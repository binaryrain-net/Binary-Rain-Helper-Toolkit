"""Tests for redirecting Spark writes and table reads to the local cache."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow.dataset as ds
import pytest
from pyspark.errors import AnalysisException
from pyspark.sql import DataFrame
from pyspark.sql.readwriter import DataFrameWriter as OriginalDataFrameWriter

from binaryrain_fabric_local_pyspark import schemas as schemas_module
from binaryrain_fabric_local_pyspark.exceptions import FabricWriteError
from binaryrain_fabric_local_pyspark.spark import DataFrameWriter, SparkSession

ABFSS = "abfss://ws@onelake.dfs.fabric.microsoft.com/lh/Tables/dbo/orders"


@pytest.fixture(autouse=True)
def cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Configure the default lakehouse and point the cache at a temporary folder."""
    monkeypatch.setenv("FABRIC_WORKSPACE_ID", "ws")
    monkeypatch.setenv("FABRIC_LAKEHOUSE_ID", "lh")
    monkeypatch.delenv("FABRIC_CACHE_REFRESH", raising=False)
    directory = tmp_path / "cache"
    monkeypatch.setattr(schemas_module, "CACHE_DIR", directory)
    return directory


@pytest.fixture(autouse=True)
def drop_temp_views(spark: SparkSession) -> Iterator[None]:
    """Remove temp views created by a test."""
    yield
    for table in spark.catalog.listTables():
        if table.isTemporary:
            spark.catalog.dropTempView(table.name)


def _frame(spark: SparkSession, count: int = 3) -> DataFrame:
    """Build a DataFrame with an id and a region column."""
    return spark.range(count).selectExpr("id", "IF(id % 2 = 0, 'even', 'odd') AS region")


def _ids(frame: DataFrame) -> list[int]:
    """Return the sorted ids of a DataFrame."""
    return sorted(row.id for row in frame.collect())


def test_dataframe_write_uses_fabric_writer(spark: SparkSession) -> None:
    """DataFrame.write returns the redirecting writer."""
    assert isinstance(_frame(spark).write, DataFrameWriter)


def test_save_as_table_writes_to_cache(spark: SparkSession, cache_dir: Path) -> None:
    """The Fabric write chain from notebooks is stored in the cache, not in Fabric."""
    _frame(spark).write.mode("overwrite").option("overwriteSchema", "true").format(
        "delta"
    ).saveAsTable("lh_gold.dbo.customers")

    target = cache_dir / "ws" / "lh_gold.Lakehouse" / "Tables" / "dbo" / "customers"
    assert sorted(ds.dataset(target).to_table()["id"].to_pylist()) == [0, 1, 2]


def test_saved_table_is_read_back(spark: SparkSession) -> None:
    """Tables written locally are read back with spark.table and spark.read.table."""
    _frame(spark).write.saveAsTable("lh_gold.dbo.customers")

    assert _ids(spark.table("lh_gold.dbo.customers")) == [0, 1, 2]
    assert _ids(spark.read.table("lh_gold.dbo.customers")) == [0, 1, 2]


def test_default_lakehouse_table_is_read_back(spark: SparkSession) -> None:
    """One and two part names use the default lakehouse and the dbo schema."""
    _frame(spark).write.saveAsTable("customers")

    assert _ids(spark.table("dbo.customers")) == [0, 1, 2]


def test_save_as_table_arguments_override_writer_settings(spark: SparkSession) -> None:
    """Mode, format and partition arguments of saveAsTable take precedence."""
    _frame(spark).write.saveAsTable("customers")

    _frame(spark, 1).write.mode("error").format("csv").saveAsTable(
        "customers", format="parquet", mode="overwrite", partitionBy="region"
    )

    assert _ids(spark.table("customers")) == [0]


def test_existing_table_needs_a_save_mode(spark: SparkSession) -> None:
    """Writing an existing table without a mode fails like in Spark."""
    _frame(spark).write.saveAsTable("customers")

    with pytest.raises(FabricWriteError, match="already exists"):
        _frame(spark).write.saveAsTable("customers")


def test_append_adds_rows(spark: SparkSession) -> None:
    """Append mode adds rows to the cached table."""
    _frame(spark, 2).write.saveAsTable("customers")

    _frame(spark, 1).write.mode("append").saveAsTable("customers")

    assert _ids(spark.table("customers")) == [0, 0, 1]


def test_append_with_merge_schema_reads_all_columns(spark: SparkSession) -> None:
    """mergeSchema allows new columns, and table reads merge the parquet schemas."""
    _frame(spark, 1).write.saveAsTable("customers")

    _frame(spark, 1).selectExpr("id", "region", "true AS extra").write.mode("append").options(
        mergeSchema=True
    ).saveAsTable("customers")

    assert sorted(spark.table("customers").columns) == ["extra", "id", "region"]


def test_append_without_merge_schema_rejects_new_columns(spark: SparkSession) -> None:
    """Appending new columns fails unless mergeSchema is set."""
    _frame(spark, 1).write.saveAsTable("customers")

    with pytest.raises(FabricWriteError, match="mergeSchema"):
        _frame(spark, 1).selectExpr("id", "region", "1 AS extra").write.mode("append").saveAsTable(
            "customers"
        )


@pytest.mark.parametrize("columns", [("region",), (["region"],)])
def test_partition_by_writes_hive_folders(
    spark: SparkSession, cache_dir: Path, columns: tuple[str | list[str], ...]
) -> None:
    """partitionBy accepts column names or a list and writes partition folders."""
    _frame(spark).write.partitionBy(*columns).saveAsTable("customers")

    target = cache_dir / "ws" / "lh" / "Tables" / "dbo" / "customers"
    assert sorted(path.name for path in target.iterdir()) == ["region=even", "region=odd"]
    assert sorted(spark.table("customers").columns) == ["id", "region"]


def test_mode_none_keeps_previous_mode(spark: SparkSession) -> None:
    """mode(None) does not reset an earlier mode."""
    _frame(spark).write.saveAsTable("customers")

    _frame(spark, 1).write.mode("overwrite").mode(None).saveAsTable("customers")

    assert _ids(spark.table("customers")) == [0]


def test_unsupported_format_raises(spark: SparkSession) -> None:
    """Only delta and parquet writes are redirected."""
    with pytest.raises(FabricWriteError, match="'csv'"):
        _frame(spark).write.format("csv").saveAsTable("customers")


@pytest.mark.parametrize("path", [ABFSS, "Tables/dbo/orders"])
def test_save_to_fabric_path_writes_to_cache(spark: SparkSession, path: str) -> None:
    """save with abfss or default lakehouse paths writes to the cache."""
    _frame(spark).write.format("delta").save(path, mode="overwrite")

    assert _ids(spark.table("dbo.orders")) == [0, 1, 2]


def test_parquet_to_fabric_path_writes_to_cache(spark: SparkSession, cache_dir: Path) -> None:
    """parquet with a Files path writes to the cache and is read back through the reader."""
    _frame(spark).write.parquet("Files/exports/customers", mode="overwrite")

    assert (cache_dir / "ws" / "lh" / "Files" / "exports" / "customers").is_dir()
    assert _ids(spark.read.parquet("Files/exports/customers")) == [0, 1, 2]


def test_cached_partitioned_folder_is_read_with_partition_columns(spark: SparkSession) -> None:
    """Cached folders are read file by file and keep their partition columns."""
    _frame(spark).write.partitionBy("region").saveAsTable("orders")
    (spark.range(0).selectExpr("id", "'none' AS region")).write.mode("append").partitionBy(
        "region"
    ).saveAsTable("orders")

    result = spark.read.parquet("Tables/dbo/orders")

    assert sorted((row.id, row.region) for row in result.collect()) == [
        (0, "even"),
        (1, "odd"),
        (2, "even"),
    ]


def test_cached_folder_ignores_hidden_files(spark: SparkSession, cache_dir: Path) -> None:
    """Files and folders starting with an underscore or a dot are skipped like in Spark."""
    _frame(spark).write.saveAsTable("orders")
    target = cache_dir / "ws" / "lh" / "Tables" / "dbo" / "orders"
    (target / "_delta_log").mkdir()
    (target / "_delta_log" / "x.parquet").write_bytes(b"not parquet")
    (target / ".hidden.parquet").write_bytes(b"not parquet")

    assert _ids(spark.read.parquet("Tables/dbo/orders")) == [0, 1, 2]


def test_save_to_local_path_uses_spark(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """save with a local path is passed to Spark unchanged."""
    save = MagicMock()
    monkeypatch.setattr(OriginalDataFrameWriter, "save", save)
    path = str(tmp_path / "out")

    _frame(spark).write.save(path, format="parquet", mode="overwrite", compression="snappy")

    save.assert_called_once_with(path, "parquet", "overwrite", None, compression="snappy")


def test_save_without_path_uses_spark(spark: SparkSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """save without a path is passed to Spark unchanged."""
    save = MagicMock()
    monkeypatch.setattr(OriginalDataFrameWriter, "save", save)

    _frame(spark).write.save()

    save.assert_called_once_with(None, None, None, None)


def test_parquet_to_local_path_uses_spark(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """parquet with a local path is passed to Spark unchanged."""
    parquet = MagicMock()
    monkeypatch.setattr(OriginalDataFrameWriter, "parquet", parquet)
    path = str(tmp_path / "out")

    _frame(spark).write.parquet(path, "overwrite", "region", "snappy")

    parquet.assert_called_once_with(path, "overwrite", "region", "snappy")


def test_table_reads_temp_views(spark: SparkSession) -> None:
    """Temp views are still read by spark.table."""
    _frame(spark).createOrReplaceTempView("customers")

    assert _ids(spark.table("customers")) == [0, 1, 2]


@pytest.mark.parametrize("name", ["missing", "lh_gold.dbo.missing"])
def test_unknown_table_raises(spark: SparkSession, name: str) -> None:
    """Names that are neither views nor cached raise the Spark error."""
    with pytest.raises(AnalysisException):
        spark.table(name)


def test_table_without_fabric_ids_raises(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without default Fabric IDs, unknown short names raise the Spark error."""
    monkeypatch.delenv("FABRIC_WORKSPACE_ID")

    with pytest.raises(AnalysisException, match="TABLE_OR_VIEW_NOT_FOUND"):
        spark.table("customers")


def test_table_with_too_many_parts_raises(spark: SparkSession) -> None:
    """Names with more than four parts raise the Spark error."""
    with pytest.raises(AnalysisException):
        spark.table("a.b.c.d.e")


def test_table_with_invalid_syntax_raises(spark: SparkSession) -> None:
    """Spark errors other than a missing table are raised unchanged."""
    with pytest.raises(AnalysisException) as error:
        spark.table("not a name")

    assert error.value.getCondition() not in {
        "TABLE_OR_VIEW_NOT_FOUND",
        "REQUIRES_SINGLE_PART_NAMESPACE",
    }
