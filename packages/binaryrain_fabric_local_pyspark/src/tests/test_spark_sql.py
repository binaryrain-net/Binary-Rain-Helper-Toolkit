"""Tests for SQL routing between local Spark temp views and the Fabric endpoint."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pyspark.errors import AnalysisException

from binaryrain_fabric_local_pyspark import spark as spark_module
from binaryrain_fabric_local_pyspark.exceptions import FabricCacheError
from binaryrain_fabric_local_pyspark.spark import SparkSession


@pytest.fixture(autouse=True)
def drop_temp_views(spark: SparkSession) -> Iterator[None]:
    """Remove temp views created by a test."""
    yield
    for table in spark.catalog.listTables():
        if table.isTemporary:
            spark.catalog.dropTempView(table.name)


@pytest.fixture
def fabric_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MagicMock:
    """Replace the Fabric endpoint with a mock and point the cache at a temp folder."""
    fabric_result = pd.DataFrame({"source": ["fabric"]})
    mock = MagicMock(return_value=fabric_result)
    monkeypatch.setattr(spark_module, "get_fabric_table_from_lakehouse", mock)
    monkeypatch.setattr(spark_module, "get_fabric_engine", MagicMock())
    monkeypatch.setattr(spark_module, "CACHE_DIR", tmp_path / "cache" / "files")
    monkeypatch.delenv("FABRIC_CACHE_REFRESH", raising=False)
    return mock


def _create_regio_view(spark: SparkSession) -> None:
    """Register a small REGIO_MAPPING temp view."""
    spark.range(1, 3).selectExpr(
        "CAST(id AS INT) AS Kundennummer",
        "CASE WHEN id = 1 THEN 'R1' END AS InsightHealthRegionalID",
    ).createOrReplaceTempView("REGIO_MAPPING")


def test_temp_view_query_runs_locally(spark: SparkSession, fabric_mock: MagicMock) -> None:
    """A query on a temp view runs on local Spark without calling Fabric."""
    _create_regio_view(spark)

    result = spark.sql(
        """
        SELECT Kundennummer, InsightHealthRegionalID
        FROM REGIO_MAPPING
        WHERE InsightHealthRegionalID IS NOT NULL
        """
    ).collect()

    assert [tuple(row) for row in result] == [(1, "R1")]
    fabric_mock.assert_not_called()


def test_query_without_temp_views_uses_fabric(spark: SparkSession, fabric_mock: MagicMock) -> None:
    """Without temp views, the query runs through the Fabric endpoint and is cached."""
    result = spark.sql("SELECT * FROM dbo.Kunden").collect()

    assert [row.source for row in result] == ["fabric"]
    fabric_mock.assert_called_once()


def test_tsql_syntax_falls_back_to_fabric(spark: SparkSession, fabric_mock: MagicMock) -> None:
    """T-SQL that Spark cannot parse falls back to Fabric when temp views exist."""
    _create_regio_view(spark)

    result = spark.sql("SELECT TOP 1 * FROM dbo.Kunden").collect()

    assert [row.source for row in result] == ["fabric"]
    fabric_mock.assert_called_once()


def test_unknown_table_falls_back_to_fabric(spark: SparkSession, fabric_mock: MagicMock) -> None:
    """A table that is not a temp view falls back to Fabric when temp views exist."""
    _create_regio_view(spark)

    result = spark.sql("SELECT * FROM Kunden").collect()

    assert [row.source for row in result] == ["fabric"]
    fabric_mock.assert_called_once()


def test_three_part_name_falls_back_to_fabric(spark: SparkSession, fabric_mock: MagicMock) -> None:
    """A lakehouse.schema.table name falls back to Fabric when temp views exist."""
    _create_regio_view(spark)

    result = spark.sql("SELECT * FROM lh_bronze.dbo.Kunden").collect()

    assert [row.source for row in result] == ["fabric"]
    fabric_mock.assert_called_once()


def test_unknown_column_on_temp_view_raises(spark: SparkSession, fabric_mock: MagicMock) -> None:
    """Errors other than parse or missing table errors are raised, not sent to Fabric."""
    _create_regio_view(spark)

    with pytest.raises(AnalysisException):
        spark.sql("SELECT MissingColumn FROM REGIO_MAPPING")

    fabric_mock.assert_not_called()


def test_query_with_args_runs_locally(spark: SparkSession, fabric_mock: MagicMock) -> None:
    """Parameterized queries run on local Spark."""
    result = spark.sql("SELECT :value AS value", args={"value": 5}).collect()

    assert result[0].value == 5
    fabric_mock.assert_not_called()


def test_fabric_query_is_served_from_cache(spark: SparkSession, fabric_mock: MagicMock) -> None:
    """Running the same Fabric query twice only calls the endpoint once."""
    spark.sql("SELECT * FROM dbo.Kunden").collect()
    result = spark.sql("  SELECT * FROM dbo.Kunden  ").collect()

    assert [row.source for row in result] == ["fabric"]
    fabric_mock.assert_called_once()


def test_cache_refresh_queries_fabric_again(
    spark: SparkSession, fabric_mock: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FABRIC_CACHE_REFRESH=1 bypasses the cached query result."""
    spark.sql("SELECT * FROM dbo.Kunden").collect()
    monkeypatch.setenv("FABRIC_CACHE_REFRESH", "1")
    spark.sql("SELECT * FROM dbo.Kunden").collect()

    assert fabric_mock.call_count == 2


def test_different_servers_use_different_cache_entries(
    spark: SparkSession, fabric_mock: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache key includes the SQL endpoint."""
    monkeypatch.setenv("FABRIC_SERVER", "server-a")
    spark.sql("SELECT * FROM dbo.Kunden").collect()
    monkeypatch.setenv("FABRIC_SERVER", "server-b")
    spark.sql("SELECT * FROM dbo.Kunden").collect()

    assert fabric_mock.call_count == 2


def test_cache_write_error_raises_cache_error(
    spark: SparkSession, fabric_mock: MagicMock, tmp_path: Path
) -> None:
    """A failure while caching the result raises FabricCacheError and removes the temp file."""
    result = MagicMock()
    result.to_parquet.side_effect = OSError("disk full")
    fabric_mock.return_value = result

    with pytest.raises(FabricCacheError, match="disk full"):
        spark.sql("SELECT * FROM dbo.Kunden")

    assert not list((tmp_path / "cache" / "sql").glob("*.tmp"))
