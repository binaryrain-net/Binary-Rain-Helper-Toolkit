"""Provide local Spark readers and SQL execution with Fabric-backed caching."""

import hashlib
import logging
import os
import subprocess
from functools import partial
from typing import Any

from pyspark import SparkConf, SparkContext, java_gateway
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession as OriginalSparkSession
from pyspark.sql.readwriter import DataFrameReader as OriginalDataFrameReader

from binaryrain_helper_local_pyspark.config import CACHE_DIR
from binaryrain_helper_local_pyspark.fabric import resolve_local_path
from binaryrain_helper_local_pyspark.sql import get_fabric_engine, get_fabric_table_from_lakehouse

logger = logging.getLogger(__name__)


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
        exception : ValueError
            If a default lakehouse-relative path is used without the required Fabric IDs.
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
        Run a SQL query through the Fabric endpoint and cache its result, or use local Spark.

        The cache key is a SHA-256 hash of the endpoint and the query text, so the same
        query maps to the same parquet file. Non-empty ``args`` or ``kwargs`` bypass the
        Fabric endpoint and cache, and are passed to the base Spark SQL implementation.

        :param str sqlQuery:
            T-SQL query to run through the Fabric SQL endpoint when no arguments are supplied.
        :param dict[str, Any] | list[Any] | None args:
            Spark SQL parameters. Non-empty values select local Spark execution. Default is None.
        :param kwargs:
            Keyword arguments passed to the base Spark SQL implementation. Non-empty values
            select local Spark execution.

        :returns DataFrame:
            The query result, loaded from the cache for Fabric endpoint queries.
        exception : ValueError
            If no server is configured when the SQL engine is first created.
        """
        if args or kwargs:
            return super().sql(sqlQuery, args, **kwargs)

        server = os.getenv("FABRIC_SERVER", "")
        cache_id = hashlib.sha256(f"{server}\n{sqlQuery.strip()}".encode()).hexdigest()
        target = CACHE_DIR.parent / "sql" / f"{cache_id}.parquet"

        if target.exists() and os.getenv("FABRIC_CACHE_REFRESH") != "1":
            logger.info("Cache hit, reading local query result: %s", target)
        else:
            logger.info("Cache miss, querying Fabric SQL endpoint: %s", cache_id)
            result = get_fabric_table_from_lakehouse(get_fabric_engine(), sqlQuery)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = target.with_name(f"{target.name}.tmp")
            # Spark cannot read nanosecond timestamps, which pandas writes by default.
            result.to_parquet(
                tmp_path,
                index=False,
                coerce_timestamps="us",
                allow_truncated_timestamps=True,
            )
            tmp_path.replace(target)

        return self.read.parquet(str(target))


def get_local_spark() -> SparkSession:
    """
    Create or reuse a local Spark session with the Fabric-aware reader installed.

    The PySpark builder creates its base session class. This function changes the resulting
    session's class so its ``read`` property uses :class:`DataFrameReader`.

    :returns SparkSession:
        Local Spark session configured to use UTC and resolve Fabric data through the cache.
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
    spark = OriginalSparkSession.builder.config(conf=conf).getOrCreate()
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
    away from the console here.

    :param SparkConf conf:
        Spark configuration used to launch the JVM.
    """
    if SparkContext._gateway is not None:
        return
    log_path = CACHE_DIR.parent / "spark.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    gateway = java_gateway.launch_gateway(
        conf, popen_kwargs={"stdout": log_file, "stderr": subprocess.STDOUT}
    )
    SparkContext._gateway = gateway
    SparkContext._jvm = gateway.jvm
    java_gateway.Popen = partial(
        subprocess.Popen, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    logger.debug("Spark JVM output is written to %s", log_path)
