"""Launch the local Spark JVM and control its console output."""

import logging
import os
import subprocess
from functools import partial

from pyspark import SparkConf, SparkContext, java_gateway
from pyspark.errors import PySparkException

from binaryrain_fabric_local_pyspark.config import CACHE_DIR
from binaryrain_fabric_local_pyspark.exceptions import SparkLaunchError

logger = logging.getLogger(__package__)


_JAVA_HINT = "Check that Java 17 or newer is installed and JAVA_HOME points to it."


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
