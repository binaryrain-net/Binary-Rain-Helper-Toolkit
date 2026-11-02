"""Tests for resolving workspace and lakehouse names to IDs."""

import io
import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError

import pytest

from binaryrain_fabric_local_pyspark import lakehouses as lakehouses_module
from binaryrain_fabric_local_pyspark.config import FABRIC_API_URL
from binaryrain_fabric_local_pyspark.exceptions import (
    FabricAuthenticationError,
    FabricConfigurationError,
    FabricDownloadError,
)
from binaryrain_fabric_local_pyspark.lakehouses import (
    clear_cache,
    is_guid,
    resolve_lakehouse_id,
    resolve_remote_path,
    resolve_workspace_id,
)
from binaryrain_fabric_local_pyspark.schemas import FabricPath

WS_ID = "11111111-1111-1111-1111-111111111111"
LH_ID = "22222222-2222-2222-2222-222222222222"
OTHER_ID = "33333333-3333-3333-3333-333333333333"
WORKSPACES_URL = f"{FABRIC_API_URL}/workspaces"
LAKEHOUSES_URL = f"{FABRIC_API_URL}/workspaces/{WS_ID}/lakehouses"


@pytest.fixture(autouse=True)
def empty_cache() -> Iterator[None]:
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lakehouses_module, "get_token", lambda scope: "token")


@pytest.fixture
def urlopen_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr(lakehouses_module, "urlopen", mock)
    return mock


def _response(body: dict[str, Any] | bytes) -> MagicMock:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    response = MagicMock()
    response.__enter__.return_value = io.BytesIO(raw)
    return response


def _serve(urlopen_mock: MagicMock, *pages: dict[str, Any] | bytes) -> None:
    urlopen_mock.side_effect = [_response(page) for page in pages]


def _item(item_id: str, name: str) -> dict[str, str]:
    return {"id": item_id, "displayName": name}


def _requested_urls(urlopen_mock: MagicMock) -> list[str]:
    return [call.args[0].full_url for call in urlopen_mock.call_args_list]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (WS_ID, True),
        ("A919C358-ABCD-4F00-9E00-0123456789AB", True),
        ("11111111111111111111111111111111", False),
        ("LH_Silver", False),
        ("", False),
    ],
)
def test_is_guid(value: str, expected: bool) -> None:
    assert is_guid(value) is expected


def test_ids_are_returned_without_request(urlopen_mock: MagicMock) -> None:
    assert resolve_workspace_id(WS_ID) == WS_ID
    assert resolve_lakehouse_id(WS_ID, LH_ID) == LH_ID
    assert resolve_lakehouse_id(WS_ID, f"{LH_ID}.Lakehouse") == LH_ID
    urlopen_mock.assert_not_called()


def test_workspace_name_is_resolved(urlopen_mock: MagicMock) -> None:
    _serve(urlopen_mock, {"value": [_item(OTHER_ID, "Other"), _item(WS_ID, "Bank DWH")]})

    assert resolve_workspace_id("Bank DWH") == WS_ID
    assert _requested_urls(urlopen_mock) == [WORKSPACES_URL]


def test_request_sends_bearer_token(urlopen_mock: MagicMock) -> None:
    _serve(urlopen_mock, {"value": [_item(WS_ID, "Bank DWH")]})

    resolve_workspace_id("Bank DWH")

    request = urlopen_mock.call_args.args[0]
    assert request.get_header("Authorization") == "Bearer token"


def test_lakehouse_suffix_is_stripped(urlopen_mock: MagicMock) -> None:
    _serve(urlopen_mock, {"value": [_item(LH_ID, "LH_Silver")]})

    assert resolve_lakehouse_id(WS_ID, "LH_Silver.Lakehouse") == LH_ID
    assert _requested_urls(urlopen_mock) == [LAKEHOUSES_URL]


def test_continuation_pages_are_followed(urlopen_mock: MagicMock) -> None:
    next_url = f"{LAKEHOUSES_URL}?continuationToken=abc"
    _serve(
        urlopen_mock,
        {"value": [_item(OTHER_ID, "LH_Bronze")], "continuationUri": next_url},
        {"value": [_item(LH_ID, "LH_Silver")]},
    )

    assert resolve_lakehouse_id(WS_ID, "LH_Silver") == LH_ID
    assert _requested_urls(urlopen_mock) == [LAKEHOUSES_URL, next_url]


def test_exact_match_wins_over_case_insensitive_match(urlopen_mock: MagicMock) -> None:
    _serve(urlopen_mock, {"value": [_item(OTHER_ID, "lh_silver"), _item(LH_ID, "LH_Silver")]})

    assert resolve_lakehouse_id(WS_ID, "LH_Silver") == LH_ID


def test_case_insensitive_match_is_used_as_fallback(urlopen_mock: MagicMock) -> None:
    _serve(urlopen_mock, {"value": [_item(LH_ID, "LH_Silver"), {"id": OTHER_ID}]})

    assert resolve_lakehouse_id(WS_ID, "lh_silver") == LH_ID


def test_ambiguous_name_raises(urlopen_mock: MagicMock) -> None:
    _serve(urlopen_mock, {"value": [_item(LH_ID, "LH_SILVER"), _item(OTHER_ID, "lh_silver")]})

    with pytest.raises(FabricConfigurationError, match="ambiguous") as exc_info:
        resolve_lakehouse_id(WS_ID, "LH_Silver")
    assert LH_ID in str(exc_info.value)
    assert OTHER_ID in str(exc_info.value)


def test_unknown_name_raises_without_refetch(urlopen_mock: MagicMock) -> None:
    _serve(urlopen_mock, {"value": [_item(LH_ID, "LH_Silver")]})

    with pytest.raises(FileNotFoundError, match="No lakehouse named 'LH_Gold'"):
        resolve_lakehouse_id(WS_ID, "LH_Gold")
    assert urlopen_mock.call_count == 1


def test_cached_listing_is_reused(urlopen_mock: MagicMock) -> None:
    _serve(urlopen_mock, {"value": [_item(LH_ID, "LH_Silver"), _item(OTHER_ID, "LH_Gold")]})

    assert resolve_lakehouse_id(WS_ID, "LH_Silver") == LH_ID
    assert resolve_lakehouse_id(WS_ID, "LH_Gold") == OTHER_ID
    assert urlopen_mock.call_count == 1


def test_cached_listing_is_refetched_on_miss(urlopen_mock: MagicMock) -> None:
    _serve(
        urlopen_mock,
        {"value": [_item(LH_ID, "LH_Silver")]},
        {"value": [_item(LH_ID, "LH_Silver"), _item(OTHER_ID, "LH_New")]},
    )

    assert resolve_lakehouse_id(WS_ID, "LH_Silver") == LH_ID
    assert resolve_lakehouse_id(WS_ID, "LH_New") == OTHER_ID
    assert urlopen_mock.call_count == 2


def test_refetched_listing_without_match_raises(urlopen_mock: MagicMock) -> None:
    _serve(
        urlopen_mock,
        {"value": [_item(LH_ID, "LH_Silver")]},
        {"value": [_item(LH_ID, "LH_Silver")]},
    )

    resolve_lakehouse_id(WS_ID, "LH_Silver")
    with pytest.raises(FileNotFoundError, match="No lakehouse named 'LH_Gold'"):
        resolve_lakehouse_id(WS_ID, "LH_Gold")
    assert urlopen_mock.call_count == 2


def test_clear_cache_forces_new_request(urlopen_mock: MagicMock) -> None:
    _serve(
        urlopen_mock,
        {"value": [_item(WS_ID, "Bank DWH")]},
        {"value": [_item(WS_ID, "Bank DWH")]},
    )

    resolve_workspace_id("Bank DWH")
    clear_cache()
    resolve_workspace_id("Bank DWH")

    assert urlopen_mock.call_count == 2


@pytest.mark.parametrize("code", [401, 403])
def test_rejected_token_raises_authentication_error(urlopen_mock: MagicMock, code: int) -> None:
    urlopen_mock.side_effect = HTTPError(WORKSPACES_URL, code, "denied", hdrs=None, fp=None)

    with pytest.raises(FabricAuthenticationError, match=f"HTTP {code}"):
        resolve_workspace_id("Bank DWH")


def test_not_found_raises_file_not_found(urlopen_mock: MagicMock) -> None:
    urlopen_mock.side_effect = HTTPError(LAKEHOUSES_URL, 404, "missing", hdrs=None, fp=None)

    with pytest.raises(FileNotFoundError, match="found nothing"):
        resolve_lakehouse_id(WS_ID, "LH_Silver")


def test_other_http_error_raises_download_error(urlopen_mock: MagicMock) -> None:
    urlopen_mock.side_effect = HTTPError(WORKSPACES_URL, 500, "boom", hdrs=None, fp=None)

    with pytest.raises(FabricDownloadError, match="HTTP 500: boom"):
        resolve_workspace_id("Bank DWH")


def test_connection_error_raises_download_error(urlopen_mock: MagicMock) -> None:
    urlopen_mock.side_effect = URLError("offline")

    with pytest.raises(FabricDownloadError, match="Could not reach"):
        resolve_workspace_id("Bank DWH")


def test_invalid_json_raises_download_error(urlopen_mock: MagicMock) -> None:
    _serve(urlopen_mock, b"not json")

    with pytest.raises(FabricDownloadError, match="invalid JSON"):
        resolve_workspace_id("Bank DWH")


def test_remote_path_with_ids_is_returned_unchanged(urlopen_mock: MagicMock) -> None:
    path = FabricPath(WS_ID, LH_ID, "Tables", "dbo/kunden")

    assert resolve_remote_path(path) is path
    urlopen_mock.assert_not_called()


def test_remote_path_with_names_is_resolved(urlopen_mock: MagicMock) -> None:
    _serve(
        urlopen_mock,
        {"value": [_item(WS_ID, "Bank DWH")]},
        {"value": [_item(LH_ID, "LH_Silver")]},
    )
    path = FabricPath("Bank DWH", "LH_Silver.Lakehouse", "Tables", "dbo/kunden")

    resolved = resolve_remote_path(path)

    assert resolved == FabricPath(WS_ID, LH_ID, "Tables", "dbo/kunden")
    assert resolved.local_path == path.local_path.parent.parent.parent.parent.parent.joinpath(
        WS_ID, LH_ID, "Tables", "dbo", "kunden"
    )
    assert _requested_urls(urlopen_mock) == [WORKSPACES_URL, LAKEHOUSES_URL]
