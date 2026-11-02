"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest

from binaryrain_fabric_local_pyspark import fabric as fabric_module
from binaryrain_fabric_local_pyspark.spark import SparkSession, get_local_spark


@pytest.fixture(autouse=True)
def skip_name_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep workspace and lakehouse names as they are, so no test calls the Fabric API."""
    monkeypatch.setattr(fabric_module, "resolve_remote_path", lambda path: path)


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """Provide one local Spark session for all tests."""
    session = get_local_spark()
    yield session
    session.stop()
