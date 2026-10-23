"""Provide local Spark readers and SQL execution with Fabric-backed caching."""

import hashlib
import logging
import os
import subprocess
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow as pa
from pyspark import SparkConf, SparkContext, java_gateway
from pyspark.errors import AnalysisException, PySparkException
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession as OriginalSparkSession
from pyspark.sql.classic.dataframe import DataFrame as ClassicDataFrame
from pyspark.sql.readwriter import DataFrameReader as OriginalDataFrameReader
from pyspark.sql.readwriter import DataFrameWriter as OriginalDataFrameWriter
from pyspark.sql.utils import to_str

from binaryrain_fabric_local_pyspark.config import CACHE_DIR
from binaryrain_fabric_local_pyspark.exceptions import (
    FabricCacheError,
    FabricConfigurationError,
    FabricWriteError,
    SparkLaunchError,
)
from binaryrain_fabric_local_pyspark.fabric import (
    parse_fabric_path,
    parse_table_name,
    resolve_local_path,
    write_cache,
)
from binaryrain_fabric_local_pyspark.schemas import FabricPath
from binaryrain_fabric_local_pyspark.sql import get_fabric_engine, get_fabric_table_from_lakehouse

logger = logging.getLogger(__name__)

_JAVA_HINT = "Check that Java 17 or newer is installed and JAVA_HOME points to it."

# Local Spark errors that mean the query targets the Fabric endpoint, not a temp view.
_FABRIC_FALLBACK_CONDITIONS = frozenset(
    {"PARSE_SYNTAX_ERROR", "TABLE_OR_VIEW_NOT_FOUND", "REQUIRES_SINGLE_PART_NAMESPACE"}
)
# Local Spark errors that mean a table name is not a temp view and may be in the cache.
_TABLE_FALLBACK_CONDITIONS = frozenset(
    {"TABLE_OR_VIEW_NOT_FOUND", "REQUIRES_SINGLE_PART_NAMESPACE"}
)
# Formats that can be redirected, both are written as parquet. Fabric defaults to delta.
_REDIRECT_FORMATS = frozenset({"delta", "parquet"})
_DEFAULT_WRITE_FORMAT = "delta"


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
        Read a Spark table or view, falling back to tables written to the local cache.

        Temporary views and tables known to local Spark are read first. Other names are
        resolved with ``parse_table_name`` and read from the cache, where
        ``DataFrameWriter.saveAsTable`` stores them.

        :param str tableName:
            Temp view name or table name such as ``lh_gold.dbo.customers``.

        :returns DataFrame:
            The table data.
        exception : AnalysisException
            If the name is neither known to local Spark nor in the local cache.
        """
        try:
            return super().table(tableName)
        except AnalysisException as exc:
            if exc.getCondition() not in _TABLE_FALLBACK_CONDITIONS:
                raise
            local_path = _cached_table_path(tableName)
            if local_path is None:
                raise
        logger.info("Reading table %s from the local cache: %s", tableName, local_path)
        # Appends with mergeSchema can leave parquet files with different columns.
        return self._read_cached(local_path, mergeSchema="true")


class DataFrameWriter(OriginalDataFrameWriter):
    """
    Redirect writes to Fabric lakehouses into the local cache instead of OneLake.

    ``saveAsTable`` always writes to the cache. ``save`` and ``parquet`` write to the cache
    when the path is an abfss URI or a default lakehouse relative path, and to the given
    local path otherwise. Delta and parquet writes are stored as parquet files, so no Delta
    Lake or Hadoop setup is needed and production data cannot be overwritten.
    """

    def __init__(self, df: DataFrame) -> None:
        """
        Create a writer for ``df`` that records its settings for redirected writes.

        :param DataFrame df:
            DataFrame to write.
        """
        super().__init__(df)
        self._save_mode: str | None = None
        self._source: str | None = None
        self._write_options: dict[str, str | None] = {}
        self._partition_columns: list[str] = []

    def mode(self, saveMode: str | None) -> "DataFrameWriter":
        """
        Set the save mode, see :meth:`pyspark.sql.DataFrameWriter.mode`.

        :param str | None saveMode:
            Spark save mode. None keeps the current mode.

        :returns DataFrameWriter:
            This writer.
        """
        if saveMode is not None:
            self._save_mode = saveMode
        super().mode(saveMode)
        return self

    def format(self, source: str) -> "DataFrameWriter":
        """
        Set the output format, see :meth:`pyspark.sql.DataFrameWriter.format`.

        :param str source:
            Output format such as ``delta`` or ``parquet``.

        :returns DataFrameWriter:
            This writer.
        """
        self._source = source
        super().format(source)
        return self

    def option(self, key: str, value: str | float | bool | None) -> "DataFrameWriter":
        """
        Add a write option, see :meth:`pyspark.sql.DataFrameWriter.option`.

        :param str key:
            Option name. Names are case insensitive.
        :param str | float | bool | None value:
            Option value.

        :returns DataFrameWriter:
            This writer.
        """
        self._write_options[key.lower()] = to_str(value)
        super().option(key, value)
        return self

    def options(self, **options: str | float | bool | None) -> "DataFrameWriter":
        """
        Add write options, see :meth:`pyspark.sql.DataFrameWriter.options`.

        :param options:
            Option names and values. Names are case insensitive.

        :returns DataFrameWriter:
            This writer.
        """
        self._write_options.update({key.lower(): to_str(value) for key, value in options.items()})
        super().options(**options)
        return self

    def partitionBy(self, *cols: str | list[str]) -> "DataFrameWriter":
        """
        Set partition columns, see :meth:`pyspark.sql.DataFrameWriter.partitionBy`.

        :param cols:
            Column names, or a single list of column names.

        :returns DataFrameWriter:
            This writer.
        """
        self._partition_columns = _column_list(cols[0] if len(cols) == 1 else list(cols))
        super().partitionBy(*cols)
        return self

    def save(
        self,
        path: str | None = None,
        format: str | None = None,
        mode: str | None = None,
        partitionBy: str | list[str] | None = None,
        **options: str | float | bool | None,
    ) -> None:
        """
        Save the DataFrame, redirecting Fabric lakehouse paths into the local cache.

        :param str | None path:
            abfss URI, default lakehouse relative path, or local path. Default is None.
        :param str | None format:
            Output format. Default is None, which uses the writer's format or ``delta``.
        :param str | None mode:
            Spark save mode. Default is None, which uses the writer's mode.
        :param str | list[str] | None partitionBy:
            Partition columns. Default is None, which uses the writer's columns.
        :param options:
            Additional write options.
        exception : FabricConfigurationError
            If a default lakehouse relative path is used without the required Fabric IDs.
        exception : FabricWriteError
            If a redirected write uses an unsupported format or conflicts with the cache.
        exception : FabricCacheError
            If the data cannot be written to the local cache.
        """
        fabric_path = parse_fabric_path(path) if path else None
        if fabric_path is None:
            super().save(path, format, mode, partitionBy, **options)
            return
        self._write_to_cache(fabric_path, format, mode, partitionBy, options)

    def saveAsTable(
        self,
        name: str,
        format: str | None = None,
        mode: str | None = None,
        partitionBy: str | list[str] | None = None,
        **options: str | float | bool | None,
    ) -> None:
        """
        Save the DataFrame as a lakehouse table in the local cache.

        The table is stored at the cache location of ``Tables/<schema>/<table>``. See
        ``parse_table_name`` for supported names. Read it back with ``spark.table``.

        :param str name:
            Table name such as ``customers``, ``dbo.customers`` or ``lh_gold.dbo.customers``.
        :param str | None format:
            Output format. Default is None, which uses the writer's format or ``delta``.
        :param str | None mode:
            Spark save mode. Default is None, which uses the writer's mode.
        :param str | list[str] | None partitionBy:
            Partition columns. Default is None, which uses the writer's columns.
        :param options:
            Additional write options.
        exception : ValueError
            If the table name is invalid.
        exception : FabricConfigurationError
            If the name relies on default Fabric IDs that are unset.
        exception : FabricWriteError
            If the format is unsupported or the write conflicts with the cache.
        exception : FabricCacheError
            If the data cannot be written to the local cache.
        """
        self._write_to_cache(parse_table_name(name), format, mode, partitionBy, options)

    def parquet(
        self,
        path: str,
        mode: str | None = None,
        partitionBy: str | list[str] | None = None,
        compression: str | None = None,
    ) -> None:
        """
        Save the DataFrame as parquet, redirecting Fabric lakehouse paths into the cache.

        :param str path:
            abfss URI, default lakehouse relative path, or local path.
        :param str | None mode:
            Spark save mode. Default is None, which uses the writer's mode.
        :param str | list[str] | None partitionBy:
            Partition columns. Default is None, which uses the writer's columns.
        :param str | None compression:
            Compression codec, ignored for redirected writes. Default is None.
        exception : FabricConfigurationError
            If a default lakehouse relative path is used without the required Fabric IDs.
        exception : FabricWriteError
            If a redirected write conflicts with the cache.
        exception : FabricCacheError
            If the data cannot be written to the local cache.
        """
        fabric_path = parse_fabric_path(path)
        if fabric_path is None:
            super().parquet(path, mode, partitionBy, compression)
            return
        self._write_to_cache(fabric_path, "parquet", mode, partitionBy, {})

    def _write_to_cache(
        self,
        fabric_path: FabricPath,
        source: str | None,
        mode: str | None,
        partition_by: str | list[str] | None,
        options: dict[str, str | float | bool | None],
    ) -> None:
        """
        Write the DataFrame to the cache location of a lakehouse path.

        Arguments override the settings made with ``mode``, ``format``, ``option`` and
        ``partitionBy``. The data is collected to the driver as an Arrow table.

        :param FabricPath fabric_path:
            Lakehouse location that would be written in Fabric.
        :param str | None source:
            Output format, or None to use the writer's format.
        :param str | None mode:
            Spark save mode, or None to use the writer's mode.
        :param str | list[str] | None partition_by:
            Partition columns, or None to use the writer's columns.
        :param dict[str, str | float | bool | None] options:
            Write options passed to the write call.
        exception : FabricWriteError
            If the format is unsupported or the write conflicts with the cache.
        exception : FabricCacheError
            If the data cannot be written to the local cache.
        """
        source = (source or self._source or _DEFAULT_WRITE_FORMAT).lower()
        if source not in _REDIRECT_FORMATS:
            raise FabricWriteError(
                f"Cannot write format {source!r} to {fabric_path.uri} locally. Only "
                f"{' and '.join(sorted(_REDIRECT_FORMATS))} writes are redirected to the cache."
            )
        write_options = {
            **self._write_options,
            **{key.lower(): to_str(value) for key, value in options.items()},
        }
        columns = _column_list(partition_by) if partition_by else self._partition_columns
        write_cache(
            fabric_path,
            self._df.toArrow(),
            mode or self._save_mode,
            columns,
            merge_schema=write_options.get("mergeschema") == "true",
        )


class SparkSession(OriginalSparkSession):
    """Use the caching reader and Fabric SQL endpoint for supported Spark operations."""

    @property
    def read(self) -> DataFrameReader:
        """Return a reader that resolves Fabric lakehouse paths through the local cache."""
        return DataFrameReader(self)

    def table(self, tableName: str) -> DataFrame:
        """
        Read a Spark table or view, falling back to tables written to the local cache.

        :param str tableName:
            Temp view name or table name such as ``lh_gold.dbo.customers``.

        :returns DataFrame:
            The table data.
        exception : AnalysisException
            If the name is neither known to local Spark nor in the local cache.
        """
        return self.read.table(tableName)

    def sql(
        self,
        sqlQuery: str,
        args: dict[str, Any] | list[Any] | None = None,
        **kwargs: Any,
    ) -> DataFrame:
        """
        Run a SQL query on local Spark or through the Fabric endpoint with caching.

        Queries are routed in this order:

        1. Non-empty ``args`` or ``kwargs`` run on local Spark.
        2. When temporary views exist, the query runs on local Spark first. If Spark cannot
           parse it or cannot find a referenced table, the query falls back to Fabric.
           Other local errors, such as an unknown column on a temp view, are raised.
        3. All other queries run through the Fabric SQL endpoint and are cached.

        The cache key is a SHA-256 hash of the endpoint and the query text, so the same
        query maps to the same parquet file. Queries that run on local Spark are not cached.

        :param str sqlQuery:
            Spark SQL query for temp views, or T-SQL query for the Fabric SQL endpoint.
        :param dict[str, Any] | list[Any] | None args:
            Spark SQL parameters. Non-empty values select local Spark execution. Default is None.
        :param kwargs:
            Keyword arguments passed to the base Spark SQL implementation. Non-empty values
            select local Spark execution.

        :returns DataFrame:
            The query result, loaded from the cache for Fabric endpoint queries.
        exception : AnalysisException
            If local Spark rejects a query for a reason other than a parse error or a
            missing table.
        exception : FabricConfigurationError
            If no Fabric SQL server is configured.
        exception : FabricAuthenticationError
            If an Azure access token cannot be acquired.
        exception : FabricQueryError
            If the Fabric SQL endpoint query fails.
        exception : FabricCacheError
            If the query result cannot be written to the local cache.
        """
        if args or kwargs:
            return super().sql(sqlQuery, args, **kwargs)

        if any(table.isTemporary for table in self.catalog.listTables()):
            try:
                return super().sql(sqlQuery)
            except AnalysisException as exc:
                if exc.getCondition() not in _FABRIC_FALLBACK_CONDITIONS:
                    raise
                logger.debug("Local Spark could not run the query, using Fabric: %s", exc)

        return self._sql_from_fabric(sqlQuery)

    def _sql_from_fabric(self, sqlQuery: str) -> DataFrame:
        """
        Run a T-SQL query through the Fabric SQL endpoint and read the cached result.

        :param str sqlQuery:
            T-SQL query to run through the Fabric SQL endpoint.

        :returns DataFrame:
            The query result, loaded from the local parquet cache.
        exception : FabricConfigurationError
            If no Fabric SQL server is configured.
        exception : FabricAuthenticationError
            If an Azure access token cannot be acquired.
        exception : FabricQueryError
            If the Fabric SQL endpoint query fails.
        exception : FabricCacheError
            If the query result cannot be written to the local cache.
        """
        server = os.getenv("FABRIC_SERVER", "")
        cache_id = hashlib.sha256(f"{server}\n{sqlQuery.strip()}".encode()).hexdigest()
        target = CACHE_DIR.parent / "sql" / f"{cache_id}.parquet"

        if target.exists() and os.getenv("FABRIC_CACHE_REFRESH") != "1":
            logger.info("Cache hit, reading local query result: %s", target)
        else:
            logger.info("Cache miss, querying Fabric SQL endpoint: %s", cache_id)
            result = get_fabric_table_from_lakehouse(get_fabric_engine(), sqlQuery)
            tmp_path = target.with_name(f"{target.name}.tmp")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Spark cannot read nanosecond timestamps, which pandas writes by default.
                result.to_parquet(
                    tmp_path,
                    index=False,
                    coerce_timestamps="us",
                    allow_truncated_timestamps=True,
                )
                tmp_path.replace(target)
            except (OSError, pa.ArrowException) as exc:
                tmp_path.unlink(missing_ok=True)
                raise FabricCacheError(
                    f"Could not cache the query result at {target}: {exc}"
                ) from exc

        return self.read.parquet(str(target))


def get_local_spark() -> SparkSession:
    """
    Create or reuse a local Spark session with the Fabric-aware reader installed.

    The PySpark builder creates its base session class. This function changes the resulting
    session's class so its ``read`` property uses :class:`DataFrameReader`, and patches
    ``DataFrame.write`` so writes use :class:`DataFrameWriter`.

    :returns SparkSession:
        Local Spark session configured to use UTC and resolve Fabric data through the cache.
    exception : SparkLaunchError
        If the Spark JVM cannot start, usually because Java is missing or ``JAVA_HOME``
        is wrong.
    """
    verbose = _spark_verbose()
    conf = (
        SparkConf()
        .setMaster("local[*]")
        .setAppName("local-fabric")
        .set("spark.sql.session.timeZone", "UTC")
        .set("spark.sql.debug.maxToStringFields", "200")
        .set("spark.ui.showConsoleProgress", str(verbose).lower())
    )
    if not verbose:
        _launch_quiet_gateway(conf)
    try:
        spark = OriginalSparkSession.builder.config(conf=conf).getOrCreate()
    except (PySparkException, OSError) as exc:
        raise SparkLaunchError(f"Could not create the Spark session: {exc}. {_JAVA_HINT}") from exc
    spark.sparkContext.setLogLevel("INFO" if verbose else "ERROR")
    spark.__class__ = SparkSession
    ClassicDataFrame.write = property(DataFrameWriter, doc=DataFrameWriter.__doc__)
    return spark


def _cached_table_path(name: str) -> Path | None:
    """
    Return the cache location of a lakehouse table name if it exists.

    :param str name:
        Table name accepted by ``parse_table_name``.

    :returns Path | None:
        The cache location, or None if the name is invalid, needs unset Fabric IDs or is
        not cached.
    """
    try:
        local_path = parse_table_name(name).local_path
    except (ValueError, FabricConfigurationError):
        return None
    return local_path if local_path.exists() else None


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


def _column_list(columns: str | list[str] | tuple[str, ...]) -> list[str]:
    """
    Normalize a column name or a sequence of column names to a list.

    :param str | list[str] | tuple[str, ...] columns:
        A single column name or a sequence of column names.

    :returns list[str]:
        The column names as a list.
    """
    return [columns] if isinstance(columns, str) else list(columns)


def _spark_verbose() -> bool:
    """
    Return whether Spark and JVM console output should be shown.

    Verbose output is enabled when the package logger is at DEBUG level or
    ``FABRIC_SPARK_DEBUG`` is set to ``1``.

    :returns bool:
        Whether verbose Spark output is wanted.
    """
    return logger.isEnabledFor(logging.DEBUG) or os.getenv("FABRIC_SPARK_DEBUG") == "1"


def _launch_quiet_gateway(conf: SparkConf) -> None:
    """
    Start the Spark JVM with its console output redirected to a log file.

    The JVM prints startup warnings before any log level can be applied, and PySpark
    runs ``taskkill`` at exit on Windows, which prints to the console. Both are routed
    away from the console here. If the log file cannot be created, the quiet launch is
    skipped and PySpark starts the JVM the normal way.

    :param SparkConf conf:
        Spark configuration used to launch the JVM.

    exception : SparkLaunchError
        If the JVM fails to start.
    """
    if SparkContext._gateway is not None:
        return
    log_path = CACHE_DIR.parent / "spark.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not create %s, JVM output will not be redirected: %s", log_path, exc)
        return
    # The JVM inherits its own copy of the handle, so the parent copy is closed after launch.
    try:
        with log_file:
            gateway = java_gateway.launch_gateway(
                conf, popen_kwargs={"stdout": log_file, "stderr": subprocess.STDOUT}
            )
    except (PySparkException, OSError) as exc:
        raise SparkLaunchError(
            f"Could not start the Spark JVM: {exc}. {_JAVA_HINT} See {log_path} for JVM output."
        ) from exc
    SparkContext._gateway = gateway
    SparkContext._jvm = gateway.jvm
    java_gateway.Popen = partial(
        subprocess.Popen, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    logger.debug("Spark JVM output is written to %s", log_path)
