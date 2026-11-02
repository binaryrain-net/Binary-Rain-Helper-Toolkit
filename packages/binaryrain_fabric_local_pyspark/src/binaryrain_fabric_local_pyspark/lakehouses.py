"""Resolve Fabric workspace and lakehouse names to their IDs through the Fabric REST API.

OneLake rejects URIs that mix IDs and friendly names, and some tenants reject friendly names
entirely. The resolver turns names into IDs right before a remote request, so the local cache
can keep the names the notebook uses.
"""

import json
import logging
from dataclasses import replace
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import UUID

from binaryrain_fabric_local_pyspark.auth import get_token
from binaryrain_fabric_local_pyspark.config import (
    FABRIC_API_SCOPE,
    FABRIC_API_TIMEOUT_SECONDS,
    FABRIC_API_URL,
    LAKEHOUSE_SUFFIX,
)
from binaryrain_fabric_local_pyspark.exceptions import (
    FabricAuthenticationError,
    FabricConfigurationError,
    FabricDownloadError,
)
from binaryrain_fabric_local_pyspark.schemas import FabricPath

logger = logging.getLogger(__name__)

HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404

# Item listings per API URL, kept for the lifetime of the process.
_listings: dict[str, list[dict[str, Any]]] = {}


def is_guid(value: str) -> bool:
    """
    Return True when ``value`` is a GUID in the hyphenated form Fabric uses.

    :param str value:
        Value to check.

    :returns bool:
        True for strings like ``a919c358-0000-0000-0000-000000000000``.
    """
    try:
        return str(UUID(value)) == value.lower()
    except ValueError:
        return False


def resolve_workspace_id(workspace: str) -> str:
    """
    Return the ID of a workspace given by ID or display name.

    :param str workspace:
        Workspace ID or display name.

    :returns str:
        The workspace ID. IDs are returned unchanged without a request.
    exception : FileNotFoundError
        If no workspace with this name is visible to the signed-in user.
    exception : FabricConfigurationError
        If several workspaces match the name.
    exception : FabricAuthenticationError
        If no token can be obtained or the Fabric API rejects it.
    exception : FabricDownloadError
        If the Fabric API request fails.
    """
    if is_guid(workspace):
        return workspace
    return _find_id(f"{FABRIC_API_URL}/workspaces", workspace, "workspace")


def resolve_lakehouse_id(workspace_id: str, lakehouse: str) -> str:
    """
    Return the ID of a lakehouse given by ID, display name or ``<name>.Lakehouse``.

    :param str workspace_id:
        ID of the workspace that contains the lakehouse.
    :param str lakehouse:
        Lakehouse ID, display name or ``<name>.Lakehouse``.

    :returns str:
        The lakehouse ID. IDs are returned unchanged without a request.
    exception : FileNotFoundError
        If the workspace has no lakehouse with this name.
    exception : FabricConfigurationError
        If several lakehouses match the name.
    exception : FabricAuthenticationError
        If no token can be obtained or the Fabric API rejects it.
    exception : FabricDownloadError
        If the Fabric API request fails.
    """
    name = lakehouse.removesuffix(LAKEHOUSE_SUFFIX)
    if is_guid(name):
        return name
    url = f"{FABRIC_API_URL}/workspaces/{quote(workspace_id)}/lakehouses"
    return _find_id(url, name, "lakehouse")


def resolve_remote_path(fabric_path: FabricPath) -> FabricPath:
    """
    Return ``fabric_path`` with its workspace and lakehouse replaced by their IDs.

    Paths that already use IDs are returned unchanged without a request.

    :param FabricPath fabric_path:
        Lakehouse location using IDs, names or a mix of both.

    :returns FabricPath:
        The same location addressed by workspace and lakehouse IDs.
    exception : FileNotFoundError
        If the workspace or lakehouse cannot be found.
    exception : FabricConfigurationError
        If a name matches several workspaces or lakehouses.
    exception : FabricAuthenticationError
        If no token can be obtained or the Fabric API rejects it.
    exception : FabricDownloadError
        If the Fabric API request fails.
    """
    workspace_id = resolve_workspace_id(fabric_path.workspace)
    lakehouse_id = resolve_lakehouse_id(workspace_id, fabric_path.lakehouse)
    if (workspace_id, lakehouse_id) == (fabric_path.workspace, fabric_path.lakehouse):
        return fabric_path
    resolved = replace(fabric_path, workspace=workspace_id, lakehouse=lakehouse_id)
    logger.info("Resolved %s to %s", fabric_path.uri, resolved.uri)
    return resolved


def clear_cache() -> None:
    """Forget all cached workspace and lakehouse listings."""
    _listings.clear()


def _find_id(url: str, name: str, kind: str) -> str:
    """
    Return the ID of the item called ``name`` in the listing at ``url``.

    An exact display name match wins over a case-insensitive one. A cached listing without a
    match is fetched again once, so items created during the session are found.

    :param str url:
        Fabric API URL that lists the items.
    :param str name:
        Display name to look for.
    :param str kind:
        Item kind used in error messages, such as ``lakehouse``.

    :returns str:
        The ID of the matching item.
    exception : FileNotFoundError
        If no item matches.
    exception : FabricConfigurationError
        If several items match.
    exception : FabricAuthenticationError
        If no token can be obtained or the Fabric API rejects it.
    exception : FabricDownloadError
        If the Fabric API request fails.
    """
    cached = url in _listings
    matches = _match(_listing(url), name)
    if not matches and cached:
        _listings.pop(url)
        matches = _match(_listing(url), name)

    if not matches:
        raise FileNotFoundError(
            f"No {kind} named {name!r} is visible at {url}. Check the name and your access."
        )
    if len(matches) > 1:
        ids = ", ".join(item["id"] for item in matches)
        raise FabricConfigurationError(
            f"The {kind} name {name!r} is ambiguous, it matches {ids}. Use the ID instead."
        )
    return matches[0]["id"]


def _match(items: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """
    Return the items whose display name equals ``name``, ignoring case when nothing is exact.

    :param list[dict[str, Any]] items:
        Items from a Fabric API listing.
    :param str name:
        Display name to look for.

    :returns list[dict[str, Any]]:
        The matching items, possibly empty.
    """
    exact = [item for item in items if item.get("displayName") == name]
    if exact:
        return exact
    folded = name.casefold()
    return [item for item in items if str(item.get("displayName", "")).casefold() == folded]


def _listing(url: str) -> list[dict[str, Any]]:
    """
    Return all items listed at ``url``, following continuation pages and caching the result.

    :param str url:
        Fabric API URL that lists the items.

    :returns list[dict[str, Any]]:
        Every item of every page.
    exception : FabricAuthenticationError
        If no token can be obtained or the Fabric API rejects it.
    exception : FabricDownloadError
        If a request fails.
    """
    if url in _listings:
        return _listings[url]

    logger.info("Listing Fabric items: %s", url)
    items: list[dict[str, Any]] = []
    next_url: str | None = url
    while next_url:
        page = _get_json(next_url)
        items.extend(page.get("value", []))
        next_url = page.get("continuationUri")
    _listings[url] = items
    return items


def _get_json(url: str) -> dict[str, Any]:
    """
    Send an authenticated GET request to the Fabric API and return the decoded JSON body.

    :param str url:
        Fabric API URL.

    :returns dict[str, Any]:
        The decoded response body.
    exception : FabricAuthenticationError
        If no token can be obtained or the Fabric API answers 401 or 403.
    exception : FileNotFoundError
        If the Fabric API answers 404, for example for an unknown workspace ID.
    exception : FabricDownloadError
        If the request fails otherwise or the body is not valid JSON.
    """
    request = Request(url, headers={"Authorization": f"Bearer {get_token(FABRIC_API_SCOPE)}"})
    try:
        with urlopen(request, timeout=FABRIC_API_TIMEOUT_SECONDS) as response:  # noqa: S310
            return json.load(response)
    except HTTPError as exc:
        if exc.code in {HTTP_UNAUTHORIZED, HTTP_FORBIDDEN}:
            raise FabricAuthenticationError(
                f"The Fabric API rejected the request to {url} with HTTP {exc.code}. Run "
                "`az login` and check that your account has at least Viewer access to the "
                "workspace."
            ) from exc
        if exc.code == HTTP_NOT_FOUND:
            raise FileNotFoundError(f"The Fabric API found nothing at {url}") from exc
        raise FabricDownloadError(
            f"The Fabric API request to {url} failed with HTTP {exc.code}: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise FabricDownloadError(f"Could not reach the Fabric API at {url}: {exc}") from exc
    except ValueError as exc:
        raise FabricDownloadError(f"The Fabric API returned invalid JSON for {url}") from exc
