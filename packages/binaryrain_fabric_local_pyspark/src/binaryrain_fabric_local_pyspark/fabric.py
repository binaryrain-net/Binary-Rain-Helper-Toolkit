"""Parse Fabric lakehouse paths and cache their data on the local filesystem."""

import logging
import os
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import pyarrow.dataset as ds
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.filedatalake import DataLakeServiceClient
from deltalake import DeltaTable

from binaryrain_helper_local_pyspark.auth import _credential, _storage_token
from binaryrain_helper_local_pyspark.config import (
    LAKEHOUSE_AREAS,
    MIN_ABFSS_SEGMENTS,
    MIN_RELATIVE_SEGMENTS,
)
from binaryrain_helper_local_pyspark.schemas import FabricPath

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
        If an abfss URI is malformed or the default lakehouse IDs are missing for a relative path.
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
        raise ValueError(
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
        If a default lakehouse-relative path is used but its workspace or lakehouse ID is unset.
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
    exception : deltalake.exceptions.TableNotFoundError
        If the remote Delta table does not exist.
    """
    fabric_path = _require_area(table, "Tables")
    target = fabric_path.local_path
    if target.exists() and not refresh:
        logger.info("Cache hit, reading local copy: %s", target)
        return target

    logger.info("Cache miss, downloading from Fabric: %s", fabric_path.uri)
    options = {"bearer_token": _storage_token(), "use_fabric_endpoint": "true"}
    dataset = DeltaTable(fabric_path.uri, storage_options=options).to_pyarrow_dataset()

    tmp_path = target.with_name(f"{target.name}.tmp")
    shutil.rmtree(tmp_path, ignore_errors=True)
    ds.write_dataset(dataset, tmp_path, format="parquet")
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
    exception : FileNotFoundError
        If the remote file or folder does not exist.
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

    try:
        properties = file_system.get_file_client(remote_root).get_file_properties()
    except ResourceNotFoundError as exc:
        raise FileNotFoundError(f"{exc.error_code}: {fabric_path.uri} not found") from exc

    # Spark writes "x.parquet" as a folder, so both layouts must be supported.
    is_folder = properties.metadata.get("hdi_isfolder") == "true"
    if is_folder:
        remote_files = [
            item.name for item in file_system.get_paths(remote_root) if not item.is_directory
        ]
    else:
        remote_files = [remote_root]

    tmp_path = target.with_name(f"{target.name}.tmp")
    shutil.rmtree(tmp_path, ignore_errors=True)
    for remote_file in remote_files:
        local_file = tmp_path
        if is_folder:
            local_file = tmp_path / PurePosixPath(remote_file).relative_to(remote_root)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        with local_file.open("wb") as handle:
            file_system.get_file_client(remote_file).download_file().readinto(handle)

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
        If the path is not a Fabric path or points into another area.
    """
    fabric_path = path if isinstance(path, FabricPath) else parse_fabric_path(path)
    if fabric_path is None or fabric_path.area != area:
        raise ValueError(f"Expected a lakehouse {area} path, got {path}")
    return fabric_path


def _commit(tmp_path: Path, target: Path) -> None:
    """
    Replace the target with a completed download from a sibling temporary path.

    :param Path tmp_path:
        Temporary location holding the completed download.
    :param Path target:
        Final cache location.
    """
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.rename(target)
