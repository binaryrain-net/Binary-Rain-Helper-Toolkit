"""Tests for the Fabric aware reader and the local Spark session setup."""

import subprocess
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pyspark import SparkContext

from binaryrain_fabric_local_pyspark import spark as spark_module
from binaryrain_fabric_local_pyspark.exceptions import SparkLaunchError
from binaryrain_fabric_local_pyspark.spark import (
    DataFrameReader,
    SparkSession,
    _launch_quiet_gateway,
    _spark_verbose,
    get_local_spark,
)


@pytest.fixture
def launch_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MagicMock:
    """Pretend no JVM runs yet and replace the gateway launcher with a mock."""
    monkeypatch.setattr(SparkContext, "_gateway", None)
    monkeypatch.setattr(SparkContext, "_jvm", None)
    monkeypatch.setattr(spark_module.java_gateway, "Popen", subprocess.Popen)
    monkeypatch.setattr(spark_module, "CACHE_DIR", tmp_path / "fabric")
    mock = MagicMock()
    monkeypatch.setattr(spark_module.java_gateway, "launch_gateway", mock)
    return mock


@pytest.fixture
def builder_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the PySpark session builder and the quiet launch with mocks."""
    session_class = MagicMock()
    monkeypatch.setattr(spark_module, "OriginalSparkSession", session_class)
    monkeypatch.setattr(spark_module, "_launch_quiet_gateway", MagicMock())
    monkeypatch.delenv("FABRIC_SPARK_DEBUG", raising=False)
    return session_class.builder.config.return_value.getOrCreate


# DataFrameReader


def test_session_read_returns_fabric_reader(spark: SparkSession) -> None:
    """The session's read property returns the caching reader."""
    assert isinstance(spark.read, DataFrameReader)


def test_reader_resolves_paths_before_reading(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fabric paths are resolved to local cache paths before Spark reads them."""
    local_file = tmp_path / "kunden.parquet"
    pd.DataFrame({"id": [1, 2]}).to_parquet(local_file, index=False)
    resolve = MagicMock(return_value=str(local_file))
    monkeypatch.setattr(spark_module, "resolve_local_path", resolve)

    result = spark.read.parquet("Files/raw/kunden.parquet", mergeSchema=True)

    assert sorted(row.id for row in result.collect()) == [1, 2]
    resolve.assert_called_once_with("Files/raw/kunden.parquet")


# get_local_spark


def test_get_local_spark_returns_fabric_session(spark: SparkSession) -> None:
    """The created session uses the Fabric aware class and UTC."""
    assert isinstance(spark, SparkSession)
    assert spark.conf.get("spark.sql.session.timeZone") == "UTC"


def test_get_local_spark_quiet_by_default(builder_mock: MagicMock) -> None:
    """Without debug output the JVM is launched quietly and logs only errors."""
    session = get_local_spark()

    spark_module._launch_quiet_gateway.assert_called_once()
    session.sparkContext.setLogLevel.assert_called_once_with("ERROR")


def test_get_local_spark_verbose(builder_mock: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """FABRIC_SPARK_DEBUG=1 skips the quiet launch and logs at INFO level."""
    monkeypatch.setenv("FABRIC_SPARK_DEBUG", "1")

    session = get_local_spark()

    spark_module._launch_quiet_gateway.assert_not_called()
    session.sparkContext.setLogLevel.assert_called_once_with("INFO")


def test_get_local_spark_wraps_launch_errors(builder_mock: MagicMock) -> None:
    """Errors while creating the session raise SparkLaunchError with a Java hint."""
    builder_mock.side_effect = OSError("java not found")

    with pytest.raises(SparkLaunchError, match="JAVA_HOME"):
        get_local_spark()


# _spark_verbose


def test_spark_verbose_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """FABRIC_SPARK_DEBUG=1 enables verbose output."""
    monkeypatch.delenv("FABRIC_SPARK_DEBUG", raising=False)
    assert not _spark_verbose()

    monkeypatch.setenv("FABRIC_SPARK_DEBUG", "1")
    assert _spark_verbose()


def test_spark_verbose_from_logger(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A DEBUG level on the package logger enables verbose output."""
    monkeypatch.delenv("FABRIC_SPARK_DEBUG", raising=False)
    caplog.set_level("DEBUG", logger=spark_module.logger.name)

    assert _spark_verbose()


# _launch_quiet_gateway


def test_quiet_gateway_skips_running_jvm(
    launch_mock: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An already running JVM is reused."""
    monkeypatch.setattr(SparkContext, "_gateway", MagicMock())

    _launch_quiet_gateway(MagicMock())

    launch_mock.assert_not_called()


def test_quiet_gateway_redirects_output(launch_mock: MagicMock, tmp_path: Path) -> None:
    """The JVM writes to spark.log and later subprocesses are silenced."""
    conf = MagicMock()

    _launch_quiet_gateway(conf)

    popen_kwargs = launch_mock.call_args.kwargs["popen_kwargs"]
    assert launch_mock.call_args.args == (conf,)
    assert Path(popen_kwargs["stdout"].name) == tmp_path / "spark.log"
    assert popen_kwargs["stdout"].closed
    assert popen_kwargs["stderr"] == subprocess.STDOUT
    assert SparkContext._gateway is launch_mock.return_value
    assert SparkContext._jvm is launch_mock.return_value.jvm
    assert isinstance(spark_module.java_gateway.Popen, partial)
    assert spark_module.java_gateway.Popen.keywords == {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }


def test_quiet_gateway_without_log_file_falls_back(
    launch_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If spark.log cannot be created, the normal PySpark launch is used."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a folder")
    monkeypatch.setattr(spark_module, "CACHE_DIR", blocker / "fabric")

    _launch_quiet_gateway(MagicMock())

    launch_mock.assert_not_called()
    assert SparkContext._gateway is None
    assert "JVM output will not be redirected" in caplog.text


def test_quiet_gateway_wraps_launch_errors(launch_mock: MagicMock, tmp_path: Path) -> None:
    """A failing JVM launch raises SparkLaunchError pointing to spark.log."""
    launch_mock.side_effect = OSError("java not found")

    with pytest.raises(SparkLaunchError, match="spark.log"):
        _launch_quiet_gateway(MagicMock())

    assert SparkContext._gateway is None
