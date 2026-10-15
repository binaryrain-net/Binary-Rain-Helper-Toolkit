"""Parse Fabric lakehouse paths and cache their data on the local filesystem."""

import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from uuid import uuid4

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from azure.core.exceptions import AzureError, ClientAuthenticationError, ResourceNotFoundError
from azure.storage.filedatalake import DataLakeServiceClient
from deltalake import DeltaTable
from deltalake.exceptions import DeltaError, TableNotFoundError

from binaryrain_fabric_local_pyspark.auth import _credential, _storage_token
from binaryrain_fabric_local_pyspark.config import (
    DEFAULT_TABLE_SCHEMA,
    ERROR_SAVE_MODES,
    HIVE_DEFAULT_PARTITION,
    LAKEHOUSE_AREAS,
    MIN_ABFSS_SEGMENTS,
    MIN_RELATIVE_SEGMENTS,
    SAVE_MODES,
)
from binaryrain_fabric_local_pyspark.exceptions import (
    FabricAuthenticationError,
    FabricCacheError,
    FabricConfigurationError,
    FabricDownloadError,
    FabricWriteError,
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


def parse_table_name(name: str) -> FabricPath:
    """
    Resolve a Spark table name to the ``Tables`` location of a lakehouse.

    Names follow Fabric's schema-enabled lakehouses:

    * ``table`` and ``schema.table`` use the default lakehouse and workspace.
    * ``lakehouse.schema.table`` uses the lakehouse ``<lakehouse>.Lakehouse`` in the default
      workspace.
    * ``workspace.lakehouse.schema.table`` also names the workspace.

    A missing schema defaults to ``dbo``. Backticks around name parts are removed.

    :param str name:
        Table name with one to four dot separated parts.

    :returns FabricPath:
        The table's location below the lakehouse ``Tables`` area.
    exception : ValueError
        If the name is empty or has more than four parts.
    exception : FabricConfigurationError
        If the name relies on the default workspace or lakehouse and its ID is unset.
    """
    parts = [part.strip().strip("`") for part in name.split(".")]
    if len(parts) > 4 or not all(parts):
        raise ValueError(
            f"Expected a table name like [[workspace.]lakehouse.][schema.]table, got {name!r}"
        )

    schema = parts[-2] if len(parts) >= 2 else DEFAULT_TABLE_SCHEMA
    lakehouse = f"{parts[-3]}.Lakehouse" if len(parts) >= 3 else os.getenv("FABRIC_LAKEHOUSE_ID")
    workspace = parts[-4] if len(parts) == 4 else os.getenv("FABRIC_WORKSPACE_ID")
    if not workspace or not lakehouse:
        raise FabricConfigurationError(
            f"Table {name} uses the default lakehouse or workspace, so FABRIC_WORKSPACE_ID and "
            "FABRIC_LAKEHOUSE_ID must be set"
        )
    return FabricPath(workspace, lakehouse, "Tables", f"{schema}/{parts[-1]}")


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
        # error_code is added by the Storage SDK, not by azure-core, so it may be missing.
        error_code = getattr(exc, "error_code", None) or "ResourceNotFound"
        raise FileNotFoundError(f"{error_code}: {fabric_path.uri} not found") from exc
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


def write_cache(
    fabric_path: FabricPath,
    data: pa.Table,
    mode: str | None = None,
    partition_by: Sequence[str] = (),
    merge_schema: bool = False,
) -> Path:
    """
    Write data to the local cache location of a lakehouse path instead of to Fabric.

    The data is stored as a folder of parquet files, the same layout ``cache_table`` uses, so
    later reads of the path or table are served from the written data. Save modes follow Spark:
    ``overwrite`` replaces the cache entry, ``append`` adds a parquet file, ``ignore`` skips
    the write and ``error``, ``errorifexists`` or ``default`` fail when the entry exists.

    :param FabricPath fabric_path:
        Lakehouse location that would be written in Fabric.
    :param pa.Table data:
        Data to write.
    :param str | None mode:
        Spark save mode. Default is None, which behaves like ``errorifexists``.
    :param Sequence[str] partition_by:
        Columns used for hive style partition folders. Default is no partitioning.
    :param bool merge_schema:
        Allow appending data whose schema differs from the existing data. Default is False.

    :returns Path:
        Path to the written cache folder.
    exception : ValueError
        If the save mode is unknown.
    exception : FabricWriteError
        If the target exists and the mode does not allow writing, a partition column is
        missing, the appended schema differs, or the existing entry is a single file.
    exception : FabricCacheError
        If the data cannot be written to the cache.
    """
    mode = (mode or "errorifexists").lower()
    if mode not in SAVE_MODES:
        raise ValueError(f"Unknown save mode {mode!r}, expected one of {', '.join(SAVE_MODES)}")
    missing = [column for column in partition_by if column not in data.column_names]
    if missing:
        raise FabricWriteError(f"Partition columns {missing} are not in the data")

    target = fabric_path.local_path
    if target.exists():
        if mode in ERROR_SAVE_MODES:
            raise FabricWriteError(
                f"{fabric_path.uri} already exists in the local cache at {target}. Use "
                "mode('overwrite') or mode('append'), or delete the cache entry."
            )
        if mode == "ignore":
            logger.info("Local cache entry exists, ignoring write: %s", target)
            return target

    appending = mode == "append" and target.exists()
    if appending:
        _check_append_schema(target, data, partition_by, merge_schema)
    destination = target if appending else target.with_name(f"{target.name}.tmp")

    logger.info("Writing to the local cache instead of Fabric: %s", fabric_path.uri)
    try:
        if not appending:
            _remove(destination)
        _write_parquet(data, destination, partition_by)
    except (OSError, pa.ArrowException) as exc:
        if not appending:
            _remove(destination, quiet=True)
        raise FabricCacheError(f"Could not write the cache for {fabric_path.uri}: {exc}") from exc

    if not appending:
        _commit(destination, target)
    return target


def _write_parquet(data: pa.Table, directory: Path, partition_by: Sequence[str]) -> None:
    """
    Write data as uniquely named parquet files into a folder.

    :param pa.Table data:
        Data to write.
    :param Path directory:
        Folder that receives the files. It is created when missing.
    :param Sequence[str] partition_by:
        Columns used for hive style partition folders.
    exception : OSError
        If the files cannot be written.
    exception : pa.ArrowException
        If the data cannot be converted to parquet.
    """
    basename = f"part-{uuid4().hex}"
    if partition_by and data.num_rows:
        ds.write_dataset(
            data,
            directory,
            format="parquet",
            partitioning=list(partition_by),
            partitioning_flavor="hive",
            basename_template=f"{basename}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        return
    if partition_by:
        # Partitioning an empty table writes no files, so the schema is kept in the default
        # partition. Files beside partition folders would be skipped by Spark.
        directory = directory.joinpath(
            *(f"{column}={HIVE_DEFAULT_PARTITION}" for column in partition_by)
        )
        data = data.drop_columns(list(partition_by))
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, directory / f"{basename}.parquet")


def _check_append_schema(
    target: Path, data: pa.Table, partition_by: Sequence[str], merge_schema: bool
) -> None:
    """
    Check that appended data matches the layout and schema of the existing cache entry.

    The partition columns must match the existing partition folders. Column types are
    compared except for partition columns, whose types are inferred from folder names.
    Nullability is ignored.

    :param Path target:
        Existing cache entry.
    :param pa.Table data:
        Data to append.
    :param Sequence[str] partition_by:
        Partition columns of the write.
    :param bool merge_schema:
        Skip the column comparison because schema changes are allowed.
    exception : FabricWriteError
        If the existing entry is a single file, the partition columns differ, or the
        schemas differ.
    """
    if target.is_file():
        raise FabricWriteError(f"Cannot append to the single cached file {target}")
    existing_partitions = _partition_columns(target)
    if existing_partitions != list(partition_by):
        raise FabricWriteError(
            f"The data at {target} is partitioned by {existing_partitions}, but the append "
            f"uses {list(partition_by)}. Use the same partitionBy columns."
        )
    if merge_schema:
        return
    # Partition values live in folder names, so only the columns stored in files are compared.
    existing = {field.name: field.type for field in ds.dataset(target).schema}
    new = {field.name: field.type for field in data.schema if field.name not in partition_by}
    if existing.keys() != new.keys() or any(existing[name] != new[name] for name in new):
        raise FabricWriteError(
            f"The appended schema differs from the data at {target}. Existing: {existing}. "
            f"New: {new}. Set option('mergeSchema', 'true') to allow schema changes."
        )


def _partition_columns(directory: Path) -> list[str]:
    """
    Return the hive partition columns of a cached folder from its folder names.

    :param Path directory:
        Cached table or file folder.

    :returns list[str]:
        Partition column names from the outermost to the innermost folder level.
    """
    columns = []
    while folders := sorted(
        path
        for path in directory.iterdir()
        if path.is_dir() and "=" in path.name and not path.name.startswith(("_", "."))
    ):
        columns.append(folders[0].name.split("=", 1)[0])
        directory = folders[0]
    return columns


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
