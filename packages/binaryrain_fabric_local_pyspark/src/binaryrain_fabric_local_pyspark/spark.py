"""Provide local Spark readers and SQL execution with Fabric-backed caching."""

import hashlib
import logging
import os
import subprocess
from functools import partial
from typing import Any

import pyarrow as pa
from pyspark import SparkConf, SparkContext, java_gateway
from pyspark.errors import AnalysisException, PySparkException
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession as OriginalSparkSession
from pyspark.sql.readwriter import DataFrameReader as OriginalDataFrameReader

from binaryrain_fabric_local_pyspark.config import CACHE_DIR
from binaryrain_fabric_local_pyspark.exceptions import FabricCacheError, SparkLaunchError
from binaryrain_fabric_local_pyspark.fabric import resolve_local_path
from binaryrain_fabric_local_pyspark.sql import get_fabric_engine, get_fabric_table_from_lakehouse

logger = logging.getLogger(__name__)

_JAVA_HINT = "Check that Java 17 or newer is installed and JAVA_HOME points to it."

# Local Spark errors that mean the query targets the Fabric endpoint, not a temp view.
_FABRIC_FALLBACK_CONDITIONS = frozenset({"PARSE_SYNTAX_ERROR", "TABLE_OR_VIEW_NOT_FOUND"})


class DataFrameReader(OriginalDataFrameReader):
    """Resolve Fabric lakehouse parquet paths to local cache paths before reading."""

    def parquet(self, *paths: str, **options: str | float | bool | None) -> DataFrame:
        """
        Read parquet data, downloading Fabric lakehouse paths to the cache when needed.

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
        return super().parquet(*(resolve_local_path(path) for path in paths), **options)


class SparkSession(OriginalSparkSession):
    """Use the caching reader and Fabric SQL endpoint for supported Spark operations."""

    @property
    def read(self) -> DataFrameReader:
        """Return a reader that resolves Fabric lakehouse paths through the local cache."""
        return DataFrameReader(self)

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
    session's class so its ``read`` property uses :class:`DataFrameReader`.

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
    return spark


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
