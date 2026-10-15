"""Tests for the FabricPath data structure."""

from pathlib import Path

import pytest

from binaryrain_fabric_local_pyspark import schemas as schemas_module
from binaryrain_fabric_local_pyspark.config import ONELAKE_HOST
from binaryrain_fabric_local_pyspark.schemas import FabricPath


def test_default_host_is_onelake() -> None:
    """The host defaults to the global OneLake endpoint."""
    assert FabricPath("ws", "lh", "Files", "a.parquet").host == ONELAKE_HOST


def test_lakehouse_path() -> None:
    """lakehouse_path joins the lakehouse, area and relative path."""
    path = FabricPath("ws", "lh", "Tables", "dbo/kunden")

    assert path.lakehouse_path == "lh/Tables/dbo/kunden"


def test_uri() -> None:
    """uri builds the full abfss URI."""
    path = FabricPath("ws", "lh", "Files", "raw/a.parquet", "host.example")

    assert path.uri == "abfss://ws@host.example/lh/Files/raw/a.parquet"


def test_local_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """local_path mirrors the lakehouse layout below the cache directory."""
    monkeypatch.setattr(schemas_module, "CACHE_DIR", tmp_path)
    path = FabricPath("ws", "lh", "Files", "raw/a.parquet")

    assert path.local_path == tmp_path / "ws" / "lh" / "Files" / "raw" / "a.parquet"


def test_is_frozen() -> None:
    """FabricPath instances are immutable."""
    path = FabricPath("ws", "lh", "Files", "a.parquet")

    with pytest.raises(AttributeError):
        path.workspace = "other"  # type: ignore[misc]
