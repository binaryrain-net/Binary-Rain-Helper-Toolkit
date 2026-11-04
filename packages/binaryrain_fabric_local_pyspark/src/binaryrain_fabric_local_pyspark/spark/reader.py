"""Read Fabric data through the local Spark cache."""

import logging
from pathlib import Path
from typing import Any

from pyspark.errors import AnalysisException
from pyspark.sql import DataFrame
from pyspark.sql.readwriter import DataFrameReader as OriginalDataFrameReader

from binaryrain_fabric_local_pyspark.exceptions import FabricConfigurationError
from binaryrain_fabric_local_pyspark.fabric import (
    parse_fabric_path,
    parse_table_name,
    resolve_local_path,
)

logger = logging.getLogger(__package__)


# Local Spark errors that mean a table name is not a temp view and may be in the cache.
_TABLE_FALLBACK_CONDITIONS = frozenset(
    {"TABLE_OR_VIEW_NOT_FOUND", "REQUIRES_SINGLE_PART_NAMESPACE"}
)


class DataFrameReader(OriginalDataFrameReader):
    """Resolve Fabric lakehouse parquet paths to local cache paths before reading."""

    def parquet(self, *paths: str, **options: str | float | bool | None) -> DataFrame:
        """
        Read parquet data, downloading Fabric lakehouse paths to the cache when needed.

        A single Fabric path that is cached as a folder is read as a list of its parquet
        files with the folder as ``basePath``. Spark's folder listing needs Hadoop's native
        Windows libraries, while explicit files do not. Partition columns are still read.

        :param paths:
            One or more abfss URIs, default lakehouse-relative paths, or local paths.
        :param options:
            Keyword options passed unchanged to PySpark's parquet reader.

        :returns DataFrame:
            DataFrame loaded from the resolved local paths.
        exception : FabricConfigurationError
            If a default lakehouse-relative path is used without the required Fabric IDs.
        exception : FileNotFoundError
            If a Fabric path does not exist in the lakehouse.
        exception : FabricAuthenticationError
            If an Azure access token cannot be acquired.
        exception : FabricDownloadError
            If the data cannot be downloaded from OneLake.
        exception : FabricCacheError
            If the data cannot be written to the local cache.
        """
        resolved = [resolve_local_path(path) for path in paths]
        if len(paths) == 1 and resolved[0] != paths[0]:
            return self._read_cached(Path(resolved[0]), **options)
        return super().parquet(*resolved, **options)

    def _read_cached(self, path: Path, **options: str | float | bool | None) -> DataFrame:
        """
        Read a cached parquet file or folder without listing the folder in Spark.

        :param Path path:
            Cached parquet file or folder.
        :param options:
            Keyword options passed to PySpark's parquet reader.

        :returns DataFrame:
            DataFrame loaded from the cache.
        """
        files = _parquet_files(path) if path.is_dir() else []
        if not files:
            return super().parquet(str(path), **options)
        # parquet() ignores unknown keyword options, so basePath is set as a reader option.
        self.option("basePath", str(path))
        return super().parquet(*files, **options)

    def text(self, paths: str | list[str], **options: Any) -> DataFrame:
        """
        Read text files, downloading Fabric lakehouse ``Files`` paths to the cache when needed.

        A Fabric path that is cached as a folder is read as a list of its files, so Spark
        never lists the folder. Spark's folder listing needs Hadoop's native Windows
        libraries, while explicit files do not. Local paths are passed through unchanged.

        :param str | list[str] paths:
            One abfss URI, default lakehouse-relative path or local path, or a list of them.
        :param options:
            Keyword options passed unchanged to PySpark's text reader, such as ``wholetext``
            or ``lineSep``.

        :returns DataFrame:
            DataFrame with one ``value`` column, one row per line or per file.
        exception : FabricConfigurationError
            If a path points into the lakehouse ``Tables`` area, or if a default
            lakehouse-relative path is used without the required Fabric IDs.
        exception : FileNotFoundError
            If a Fabric path does not exist in the lakehouse.
        exception : FabricAuthenticationError
            If an Azure access token cannot be acquired.
        exception : FabricDownloadError
            If the data cannot be downloaded from OneLake.
        exception : FabricCacheError
            If the data cannot be written to the local cache.
        """
        path_list = [paths] if isinstance(paths, str) else list(paths)
        files = [file for path in path_list for file in _resolve_text_path(path)]
        return super().text(files, **options)

    def table(self, tableName: str) -> DataFrame:
        """
        Read a Spark table or view, falling back to cached or remote Fabric tables.

        Temporary views and tables known to local Spark are read first. Other names are
        resolved with ``parse_table_name`` and read through the Fabric cache. Missing
        tables are downloaded from OneLake, like ``parquet`` reads of ``Tables`` paths.

        :param str tableName:
            Temp view name or table name such as ``lh_gold.dbo.customers``.

        :returns DataFrame:
            The table data.
        exception : AnalysisException
            If local Spark cannot resolve the name and it cannot identify a Fabric table.
        exception : FileNotFoundError
            If the remote Fabric table does not exist.
        exception : FabricAuthenticationError
            If an Azure access token cannot be acquired.
        exception : FabricDownloadError
            If the table cannot be downloaded from OneLake.
        exception : FabricCacheError
            If the table cannot be written to the local cache.
        """
        try:
            return super().table(tableName)
        except AnalysisException as exc:
            if exc.getCondition() not in _TABLE_FALLBACK_CONDITIONS:
                raise
            local_path = _resolve_table_path(tableName)
            if local_path is None:
                raise
        logger.info("Reading table %s from the local cache: %s", tableName, local_path)
        # Appends with mergeSchema can leave parquet files with different columns.
        return self._read_cached(local_path, mergeSchema="true")


def _resolve_table_path(name: str) -> Path | None:
    """
    Resolve a lakehouse table name through the cache, downloading when needed.

    :param str name:
        Table name accepted by ``parse_table_name``.

    :returns Path | None:
        The cache location, or None if the name is invalid or needs unset Fabric IDs.
    exception : FileNotFoundError
        If the remote Fabric table does not exist.
    exception : FabricAuthenticationError
        If an Azure access token cannot be acquired.
    exception : FabricDownloadError
        If the table cannot be downloaded from OneLake.
    exception : FabricCacheError
        If the table cannot be written to the local cache.
    """
    try:
        fabric_path = parse_table_name(name)
    except (ValueError, FabricConfigurationError):
        return None
    return Path(resolve_local_path(fabric_path.uri))


def _resolve_text_path(path: str) -> list[str]:
    """
    Resolve one text reader path to the local files Spark should read.

    :param str path:
        An abfss URI, a default lakehouse-relative path, or a local path.

    :returns list[str]:
        The cached files of a Fabric folder, or a single path. An empty cached folder and
        any local path are returned as is.
    exception : FabricConfigurationError
        If the path points into the lakehouse ``Tables`` area.
    """
    fabric_path = parse_fabric_path(path)
    if fabric_path is not None and fabric_path.area == "Tables":
        raise FabricConfigurationError(
            f"spark.read.text reads lakehouse Files paths only, got the table path {path}"
        )
    resolved = resolve_local_path(path)
    if resolved == path or not Path(resolved).is_dir():
        return [resolved]
    return _data_files(Path(resolved)) or [resolved]


def _data_files(directory: Path, pattern: str = "*") -> list[str]:
    """
    Return the data files below a folder, skipping hidden files like Spark does.

    :param Path directory:
        Folder to search, including subfolders.
    :param str pattern:
        Glob pattern the file names must match. Default is ``*``.

    :returns list[str]:
        Sorted file paths. Paths with a part starting with ``_`` or ``.`` are skipped.
    """
    return sorted(
        str(path)
        for path in directory.rglob(pattern)
        if path.is_file()
        and not any(part.startswith(("_", ".")) for part in path.relative_to(directory).parts)
    )


def _parquet_files(directory: Path) -> list[str]:
    """
    Return the parquet data files below a folder, skipping hidden files like Spark does.

    :param Path directory:
        Folder to search, including partition subfolders.

    :returns list[str]:
        Sorted file paths. Paths with a part starting with ``_`` or ``.`` are skipped.
    """
    return _data_files(directory, "*.parquet")
