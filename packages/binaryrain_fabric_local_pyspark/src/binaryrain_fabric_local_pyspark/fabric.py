"""Parse Fabric lakehouse paths and cache their data on the local filesystem."""

import logging
import os
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.dataset as ds
from azure.core.exceptions import AzureError, ClientAuthenticationError, ResourceNotFoundError
from azure.storage.filedatalake import DataLakeServiceClient
from deltalake import DeltaTable
from deltalake.exceptions import DeltaError, TableNotFoundError

from binaryrain_fabric_local_pyspark.auth import _credential, _storage_token
from binaryrain_fabric_local_pyspark.config import (
    LAKEHOUSE_AREAS,
    MIN_ABFSS_SEGMENTS,
    MIN_RELATIVE_SEGMENTS,
)
from binaryrain_fabric_local_pyspark.exceptions import (
    FabricAuthenticationError,
    FabricCacheError,
    FabricConfigurationError,
    FabricDownloadError,
)
from binaryrain_fabric_local_pyspark.schemas import FabricPath

logger = logging.getLogger(__name__)


def parse_fabric_path(path: str) -> FabricPath | None:
    """
    Parse an abfss URI or a default lakehouse relative path into its parts.

    ``abfss://<workspace>@<host>/<lakehouse>/<Files|Tables>/<relative>`` is split directly.
    ``Files/<relative>`` and ``Tables/<relative>`` are resolved against the default lakehouse
    from ``FABRIC_WORKSPACE_ID`` and ``FABRIC_LAKEHOUSE_ID``.

    :param str path:
        An abfss URI, a default lakehouse-relative path, or a local path.

    :returns FabricPath | None:
        Parsed Fabric path, or None when the input is not a recognized Fabric path.
    exception : ValueError
        If an abfss URI is malformed.
    exception : FabricConfigurationError
        If the default lakehouse IDs are missing for a relative path.
    """
    if path.lower().startswith("abfss://"):
        parts = urlsplit(path)
        workspace, _, host = parts.netloc.partition("@")
        segments = PurePosixPath(parts.path).parts[1:]
        if (
            not workspace
            or not host
            or len(segments) < MIN_ABFSS_SEGMENTS
            or segments[1] not in LAKEHOUSE_AREAS
        ):
            raise ValueError(
                f"Expected abfss://<workspace>@<host>/<lakehouse>/<Files|Tables>/..., got {path}"
            )
        return FabricPath(workspace, segments[0], segments[1], "/".join(segments[2:]), host)

    segments = PurePosixPath(path.replace("\\", "/").lstrip("/")).parts
    if len(segments) < MIN_RELATIVE_SEGMENTS or segments[0] not in LAKEHOUSE_AREAS:
        return None

    workspace = os.getenv("FABRIC_WORKSPACE_ID")
    lakehouse = os.getenv("FABRIC_LAKEHOUSE_ID")
    if not workspace or not lakehouse:
        raise FabricConfigurationError(
            f"{path} uses the default lakehouse, so FABRIC_WORKSPACE_ID and "
            "FABRIC_LAKEHOUSE_ID must be set"
        )
    return FabricPath(workspace, lakehouse, segments[0], "/".join(segments[1:]))


def resolve_local_path(path: str, refresh: bool | None = None) -> str:
    """
    Return a local path for ``path``, downloading Fabric data into the cache when needed.

    :param str path:
        abfss URI, default lakehouse relative path, or local path.
    :param bool | None refresh:
        Refresh the cache when True or use ``FABRIC_CACHE_REFRESH`` when None. False uses the
        existing cache when available.

    :returns str:
        Local cached path for a Fabric path, or the original path for any other input.
    exception : ValueError
        If an abfss URI is malformed.
    exception : FabricConfigurationError
        If a default lakehouse-relative path is used but its workspace or lakehouse ID is unset.
    exception : FabricAuthenticationError
        If no Azure token can be obtained.
    exception : FileNotFoundError
        If the remote file, folder or table does not exist.
    exception : FabricDownloadError
        If the download from Fabric fails.
    exception : FabricCacheError
        If the local cache cannot be written.
    """
    fabric_path = parse_fabric_path(path)
    if fabric_path is None:
        return path
    if refresh is None:
        refresh = os.getenv("FABRIC_CACHE_REFRESH") == "1"
    if fabric_path.area == "Tables":
        return str(cache_table(fabric_path, refresh))
    return str(cache_file(fabric_path, refresh))


def cache_table(table: FabricPath | str, refresh: bool = False) -> Path:
    """
    Cache a lakehouse Delta table locally as parquet and return its cache directory.

    A cache hit is returned without checking the remote table unless ``refresh`` is True.
    Refreshing replaces the cached directory with data from the current table version.

    :param FabricPath | str table:
        A ``Tables`` path, such as ``Tables/dbo/kunden``, or a full abfss URI.
    :param bool refresh:
        Download again even when a cached copy exists. Default is False.

    :returns Path:
        Path to the directory containing the table's cached parquet files.
    exception : ValueError
        If the path is not a lakehouse ``Tables`` path.
    exception : FabricAuthenticationError
        If no Azure token can be obtained.
    exception : FileNotFoundError
        If the remote Delta table does not exist.
    exception : FabricDownloadError
        If reading the Delta table from Fabric fails.
    exception : FabricCacheError
        If the local cache cannot be written.
    """
    fabric_path = _require_area(table, "Tables")
    target = fabric_path.local_path
    if target.exists() and not refresh:
        logger.info("Cache hit, reading local copy: %s", target)
        return target

    logger.info("Cache miss, downloading from Fabric: %s", fabric_path.uri)
    options = {"bearer_token": _storage_token(), "use_fabric_endpoint": "true"}
    tmp_path = target.with_name(f"{target.name}.tmp")
    try:
        _remove(tmp_path)
        dataset = DeltaTable(fabric_path.uri, storage_options=options).to_pyarrow_dataset()
        ds.write_dataset(dataset, tmp_path, format="parquet")
    except TableNotFoundError as exc:
        _remove(tmp_path, quiet=True)
        raise FileNotFoundError(f"No Delta table found at {fabric_path.uri}") from exc
    except (DeltaError, pa.ArrowException) as exc:
        _remove(tmp_path, quiet=True)
        raise FabricDownloadError(
            f"Could not read the Delta table {fabric_path.uri}. Check your access to the "
            f"lakehouse and your network connection. Details: {exc}"
        ) from exc
    except OSError as exc:
        _remove(tmp_path, quiet=True)
        raise FabricCacheError(f"Could not write the cache for {fabric_path.uri}: {exc}") from exc

    _commit(tmp_path, target)
    return target


def cache_file(file: FabricPath | str, refresh: bool = False) -> Path:
    """
    Cache a file or folder below the lakehouse ``Files`` area and return its local path.

    A cache hit is returned without checking the remote item unless ``refresh`` is True.

    :param FabricPath | str file:
        A ``Files`` path, such as ``Files/onPrem/mssql/Kunden.parquet``, or a full abfss URI.
    :param bool refresh:
        Download again even when a cached copy exists. Default is False.

    :returns Path:
        Path to the cached file or folder.
    exception : ValueError
        If the path is not a lakehouse ``Files`` path.
    exception : FabricAuthenticationError
        If Fabric rejects the Azure credentials or no credential is available.
    exception : FileNotFoundError
        If the remote file or folder does not exist.
    exception : FabricDownloadError
        If the download from Fabric fails.
    exception : FabricCacheError
        If the local cache cannot be written.
    """
    fabric_path = _require_area(file, "Files")
    target = fabric_path.local_path
    if target.exists() and not refresh:
        logger.info("Cache hit, reading local copy: %s", target)
        return target

    service = DataLakeServiceClient(f"https://{fabric_path.host}", credential=_credential())
    file_system = service.get_file_system_client(fabric_path.workspace)
    remote_root = fabric_path.lakehouse_path
    logger.info("Cache miss, downloading from Fabric: %s", fabric_path.uri)

    tmp_path = target.with_name(f"{target.name}.tmp")
    try:
        properties = file_system.get_file_client(remote_root).get_file_properties()
        # Spark writes "x.parquet" as a folder, so both layouts must be supported.
        is_folder = properties.metadata.get("hdi_isfolder") == "true"
        if is_folder:
            remote_files = [
                item.name for item in file_system.get_paths(remote_root) if not item.is_directory
            ]
        else:
            remote_files = [remote_root]

        _remove(tmp_path)
        for remote_file in remote_files:
            local_file = tmp_path
            if is_folder:
                local_file = tmp_path / PurePosixPath(remote_file).relative_to(remote_root)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            with local_file.open("wb") as handle:
                file_system.get_file_client(remote_file).download_file().readinto(handle)
    except ResourceNotFoundError as exc:
        _remove(tmp_path, quiet=True)
        raise FileNotFoundError(f"{exc.error_code}: {fabric_path.uri} not found") from exc
    except ClientAuthenticationError as exc:
        _remove(tmp_path, quiet=True)
        raise FabricAuthenticationError(
            f"Fabric rejected the credentials for {fabric_path.uri}. Run `az login` and check "
            f"that your account can read the lakehouse. Details: {exc}"
        ) from exc
    except AzureError as exc:
        _remove(tmp_path, quiet=True)
        raise FabricDownloadError(f"Could not download {fabric_path.uri}: {exc}") from exc
    except OSError as exc:
        _remove(tmp_path, quiet=True)
        raise FabricCacheError(f"Could not write the cache for {fabric_path.uri}: {exc}") from exc

    _commit(tmp_path, target)
    return target


def _require_area(path: FabricPath | str, area: str) -> FabricPath:
    """
    Parse ``path`` and check that it points into the expected lakehouse area.

    :param FabricPath | str path:
        Parsed path or a string accepted by ``parse_fabric_path``.
    :param str area:
        Required lakehouse area, either ``Files`` or ``Tables``.

    :returns FabricPath:
        Parsed lakehouse path.
    exception : ValueError
        If an abfss URI is malformed.
    exception : FabricConfigurationError
        If the path is not a Fabric path or points into another area, or if the default
        lakehouse IDs are unset.
    """
    fabric_path = path if isinstance(path, FabricPath) else parse_fabric_path(path)
    if fabric_path is None or fabric_path.area != area:
        raise FabricConfigurationError(f"Expected a lakehouse {area} path, got {path}")
    return fabric_path


def _remove(path: Path, quiet: bool = False) -> None:
    """
    Delete a file or directory if it exists.

    :param Path path:
        File or directory to delete.
    :param bool quiet:
        Log a warning instead of raising when deletion fails. Used during cleanup so the
        original error is not masked. Default is False.
    exception : OSError
        If deletion fails and ``quiet`` is False.
    """
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    except OSError:
        if not quiet:
            raise
        logger.warning("Could not remove temporary cache path: %s", path, exc_info=True)


def _commit(tmp_path: Path, target: Path) -> None:
    """
    Replace the target with a completed download from a sibling temporary path.

    :param Path tmp_path:
        Temporary location holding the completed download.
    :param Path target:
        Final cache location.
    exception : FabricCacheError
        If the old cache cannot be removed or the new one cannot be moved into place.
    """
    try:
        _remove(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.rename(target)
    except OSError as exc:
        _remove(tmp_path, quiet=True)
        raise FabricCacheError(
            f"Could not move the download into the cache at {target}. Close any program using "
            f"those files and try again. Details: {exc}"
        ) from exc
