"""Tests for the public Spark package imports."""

from binaryrain_fabric_local_pyspark.spark import (
    DataFrameReader,
    DataFrameWriter,
    SparkSession,
    get_local_spark,
)
from binaryrain_fabric_local_pyspark.spark import _launch, reader, session, writer


def test_existing_spark_imports_are_reexported() -> None:
    """Existing notebook imports resolve to the implementations in the new modules."""
    assert DataFrameReader is reader.DataFrameReader
    assert DataFrameWriter is writer.DataFrameWriter
    assert SparkSession is session.SparkSession
    assert get_local_spark is session.get_local_spark


def test_spark_modules_keep_the_existing_logger_name() -> None:
    """Existing logging configuration applies to each Spark module."""
    assert reader.logger.name == "binaryrain_fabric_local_pyspark.spark"
    assert session.logger is reader.logger
    assert _launch.logger is reader.logger
