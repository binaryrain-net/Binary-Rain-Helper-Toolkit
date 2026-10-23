"""Tests for the Fabric aware reader and the local Spark session setup."""

import subprocess
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pyspark import SparkContext

from binaryrain_fabric_local_pyspark import spark as spark_module
from binaryrain_fabric_local_pyspark.exceptions import FabricConfigurationError, SparkLaunchError
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


_TEXT_URI = "abfss://ws@onelake.dfs.fabric.microsoft.com/lh/Files/raw/logs"


def test_text_passes_local_paths_through(spark: SparkSession, tmp_path: Path) -> None:
    """Local text files are read without touching the cache."""
    local_file = tmp_path / "local.txt"
    local_file.write_text("a\nb\n", encoding="utf-8")

    result = spark.read.text(str(local_file))

    assert [row.value for row in result.collect()] == ["a", "b"]


def test_text_reads_cached_file(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Fabric file path is resolved to its cached file."""
    cached = tmp_path / "file.txt"
    cached.write_text("x\ny\n", encoding="utf-8")
    resolve = MagicMock(return_value=str(cached))
    monkeypatch.setattr(spark_module, "resolve_local_path", resolve)

    result = spark.read.text(f"{_TEXT_URI}/file.txt")

    assert [row.value for row in result.collect()] == ["x", "y"]
    resolve.assert_called_once_with(f"{_TEXT_URI}/file.txt")


def test_text_expands_cached_folder(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cached folder is read file by file, skipping hidden and underscore files."""
    folder = tmp_path / "logs"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.txt").write_text("1\n", encoding="utf-8")
    (folder / "sub" / "b.log").write_text("2\n", encoding="utf-8")
    (folder / "_SUCCESS").write_text("skip\n", encoding="utf-8")
    (folder / ".hidden").write_text("skip\n", encoding="utf-8")
    monkeypatch.setattr(spark_module, "resolve_local_path", MagicMock(return_value=str(folder)))
    text_mock = MagicMock()
    monkeypatch.setattr(spark_module.OriginalDataFrameReader, "text", text_mock)

    spark.read.text(_TEXT_URI, wholetext=True)

    files = [str(folder / "a.txt"), str(folder / "sub" / "b.log")]
    text_mock.assert_called_once_with(files, wholetext=True)


def test_text_reads_cached_folder_with_spark(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Expanded folder files are read by Spark and options such as wholetext still apply."""
    folder = tmp_path / "logs"
    folder.mkdir()
    (folder / "a.txt").write_bytes(b"1\n2\n")
    (folder / "b.txt").write_bytes(b"3\n")
    monkeypatch.setattr(spark_module, "resolve_local_path", MagicMock(return_value=str(folder)))

    lines = spark.read.text(_TEXT_URI)
    whole = spark.read.text(_TEXT_URI, wholetext=True)

    assert sorted(row.value for row in lines.collect()) == ["1", "2", "3"]
    assert sorted(row.value for row in whole.collect()) == ["1\n2\n", "3\n"]


def test_text_keeps_empty_cached_folder(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty cached folder is passed to Spark as the folder itself."""
    folder = tmp_path / "empty"
    folder.mkdir()
    monkeypatch.setattr(spark_module, "resolve_local_path", MagicMock(return_value=str(folder)))
    text_mock = MagicMock()
    monkeypatch.setattr(spark_module.OriginalDataFrameReader, "text", text_mock)

    spark.read.text(_TEXT_URI)

    text_mock.assert_called_once_with([str(folder)])


def test_text_accepts_a_list_of_paths(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every path in a list is resolved and the files are read together."""
    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"
    first.write_text("one\n", encoding="utf-8")
    second.write_text("two\n", encoding="utf-8")
    cached = {f"{_TEXT_URI}/one.txt": str(first), f"{_TEXT_URI}/two.txt": str(second)}
    monkeypatch.setattr(spark_module, "resolve_local_path", cached.__getitem__)

    result = spark.read.text(list(cached))

    assert sorted(row.value for row in result.collect()) == ["one", "two"]


def test_text_rejects_table_paths(spark: SparkSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tables paths raise before anything is downloaded."""
    resolve = MagicMock()
    monkeypatch.setattr(spark_module, "resolve_local_path", resolve)

    with pytest.raises(FabricConfigurationError, match="Files paths only"):
        spark.read.text("abfss://ws@onelake.dfs.fabric.microsoft.com/lh/Tables/dbo/kunden")

    resolve.assert_not_called()


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
