"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest

from binaryrain_fabric_local_pyspark.spark import SparkSession, get_local_spark


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """Provide one local Spark session for all tests."""
    session = get_local_spark()
    yield session
    session.stop()
