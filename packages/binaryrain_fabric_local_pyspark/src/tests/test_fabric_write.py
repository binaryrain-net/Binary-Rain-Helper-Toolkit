"""Tests for resolving table names and writing data into the local cache."""

from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from binaryrain_fabric_local_pyspark import fabric as fabric_module
from binaryrain_fabric_local_pyspark import schemas as schemas_module
from binaryrain_fabric_local_pyspark.exceptions import (
    FabricCacheError,
    FabricConfigurationError,
    FabricWriteError,
)
from binaryrain_fabric_local_pyspark.fabric import parse_table_name, write_cache
from binaryrain_fabric_local_pyspark.schemas import FabricPath

TABLE = FabricPath("ws", "lh", "Tables", "dbo/customers")


@pytest.fixture(autouse=True)
def fabric_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the default lakehouse."""
    monkeypatch.setenv("FABRIC_WORKSPACE_ID", "ws")
    monkeypatch.setenv("FABRIC_LAKEHOUSE_ID", "lh")


@pytest.fixture(autouse=True)
def cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the local cache at a temporary folder."""
    directory = tmp_path / "cache"
    monkeypatch.setattr(schemas_module, "CACHE_DIR", directory)
    return directory


def _table(ids: list[int], region: str = "a") -> pa.Table:
    """Build a small table with an id and a region column."""
    return pa.table({"id": pa.array(ids, pa.int64()), "region": [region] * len(ids)})


def _read(path: Path) -> list[int]:
    """Read the sorted ids stored below a cache folder."""
    return sorted(ds.dataset(path, partitioning="hive").to_table()["id"].to_pylist())


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("customers", FabricPath("ws", "lh", "Tables", "dbo/customers")),
        ("sales.customers", FabricPath("ws", "lh", "Tables", "sales/customers")),
        ("lh_gold.dbo.customers", FabricPath("ws", "lh_gold.Lakehouse", "Tables", "dbo/customers")),
        (
            "other_ws.lh_gold.dbo.customers",
            FabricPath("other_ws", "lh_gold.Lakehouse", "Tables", "dbo/customers"),
        ),
        (
            "`lh_gold`.`dbo`.`customers`",
            FabricPath("ws", "lh_gold.Lakehouse", "Tables", "dbo/customers"),
        ),
    ],
)
def test_parse_table_name(name: str, expected: FabricPath) -> None:
    """Table names with one to four parts resolve to the lakehouse Tables area."""
    assert parse_table_name(name) == expected


@pytest.mark.parametrize("name", ["", "a..b", "a.b.c.d.e"])
def test_parse_table_name_rejects_invalid_names(name: str) -> None:
    """Empty parts and more than four parts are rejected."""
    with pytest.raises(ValueError, match="Expected a table name"):
        parse_table_name(name)


@pytest.mark.parametrize(
    ("name", "variable"),
    [
        ("customers", "FABRIC_LAKEHOUSE_ID"),
        ("dbo.customers", "FABRIC_WORKSPACE_ID"),
        ("lh_gold.dbo.customers", "FABRIC_WORKSPACE_ID"),
    ],
)
def test_parse_table_name_requires_default_ids(
    monkeypatch: pytest.MonkeyPatch, name: str, variable: str
) -> None:
    """Names that rely on the default workspace or lakehouse need their IDs."""
    monkeypatch.delenv(variable)

    with pytest.raises(FabricConfigurationError, match="FABRIC_WORKSPACE_ID"):
        parse_table_name(name)


def test_parse_table_name_with_workspace_needs_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A four part name does not use the default IDs."""
    monkeypatch.delenv("FABRIC_WORKSPACE_ID")
    monkeypatch.delenv("FABRIC_LAKEHOUSE_ID")

    assert parse_table_name("w.l.s.t") == FabricPath("w", "l.Lakehouse", "Tables", "s/t")


def test_write_creates_cache_folder() -> None:
    """A write to a new target creates a parquet folder at the cache location."""
    target = write_cache(TABLE, _table([1, 2]))

    assert target == TABLE.local_path
    assert _read(target) == [1, 2]
    assert not target.with_name(f"{target.name}.tmp").exists()


@pytest.mark.parametrize("mode", [None, "error", "errorifexists", "default", "ErrorIfExists"])
def test_error_modes_reject_existing_target(mode: str | None) -> None:
    """Error modes fail when the target already exists."""
    write_cache(TABLE, _table([1]))

    with pytest.raises(FabricWriteError, match="already exists"):
        write_cache(TABLE, _table([2]), mode)
    assert _read(TABLE.local_path) == [1]


def test_ignore_mode_keeps_existing_data() -> None:
    """Ignore mode skips the write when the target exists."""
    write_cache(TABLE, _table([1]))

    write_cache(TABLE, _table([2]), "ignore")

    assert _read(TABLE.local_path) == [1]


def test_ignore_mode_writes_new_target() -> None:
    """Ignore mode writes when the target does not exist."""
    write_cache(TABLE, _table([1]), "ignore")

    assert _read(TABLE.local_path) == [1]


def test_overwrite_replaces_data() -> None:
    """Overwrite mode replaces the existing data."""
    write_cache(TABLE, _table([1]))

    write_cache(TABLE, _table([2, 3]), "overwrite")

    assert _read(TABLE.local_path) == [2, 3]


def test_append_adds_data() -> None:
    """Append mode adds a file next to the existing data."""
    write_cache(TABLE, _table([1]))

    write_cache(TABLE, _table([2]), "append")

    assert _read(TABLE.local_path) == [1, 2]


def test_append_to_missing_target_writes_data() -> None:
    """Append mode creates the target when it does not exist."""
    write_cache(TABLE, _table([1]), "append")

    assert _read(TABLE.local_path) == [1]


@pytest.mark.parametrize(
    "data",
    [
        pa.table({"id": pa.array([2], pa.int64())}),
        pa.table({"id": pa.array([2], pa.int32()), "region": ["a"]}),
    ],
)
def test_append_rejects_schema_changes(data: pa.Table) -> None:
    """Appending different columns or types fails without mergeSchema."""
    write_cache(TABLE, _table([1]))

    with pytest.raises(FabricWriteError, match="mergeSchema"):
        write_cache(TABLE, data, "append")


def test_append_with_merge_schema_allows_new_columns() -> None:
    """Appending new columns works when mergeSchema is enabled."""
    write_cache(TABLE, _table([1]))

    write_cache(
        TABLE, _table([2]).append_column("extra", pa.array([True])), "append", merge_schema=True
    )

    assert _read(TABLE.local_path) == [1, 2]
    assert len(list(TABLE.local_path.glob("*.parquet"))) == 2


def test_append_ignores_nullability() -> None:
    """Appending the same types with different nullability is allowed."""
    write_cache(TABLE, _table([1]))
    schema = pa.schema([pa.field("id", pa.int64(), nullable=False), ("region", pa.string())])

    write_cache(TABLE, pa.table({"id": [2], "region": ["a"]}, schema=schema), "append")

    assert _read(TABLE.local_path) == [1, 2]


def test_append_to_single_file_raises() -> None:
    """A cached single file cannot be appended to."""
    target = TABLE.local_path
    target.parent.mkdir(parents=True)
    pq.write_table(_table([1]), target)

    with pytest.raises(FabricWriteError, match="single cached file"):
        write_cache(TABLE, _table([2]), "append")


def test_unknown_mode_raises() -> None:
    """Unknown save modes are rejected."""
    with pytest.raises(ValueError, match="Unknown save mode"):
        write_cache(TABLE, _table([1]), "upsert")


def test_partitioned_write_uses_hive_folders() -> None:
    """Partition columns become key=value folders."""
    data = pa.concat_tables([_table([1], "a"), _table([2], "b")])

    target = write_cache(TABLE, data, partition_by=["region"])

    assert sorted(path.name for path in target.iterdir()) == ["region=a", "region=b"]
    assert _read(target) == [1, 2]


def test_partitioned_append_keeps_existing_partitions() -> None:
    """Appending to a partitioned table keeps files in existing partition folders."""
    write_cache(TABLE, _table([1], "a"), partition_by=["region"])

    write_cache(TABLE, _table([2], "a"), "append", partition_by=["region"])

    assert _read(TABLE.local_path) == [1, 2]
    assert len(list((TABLE.local_path / "region=a").glob("*.parquet"))) == 2


def test_empty_partitioned_write_keeps_schema() -> None:
    """An empty partitioned write stores its schema in the default partition folder."""
    target = write_cache(TABLE, _table([]), partition_by=["region"])

    files = list(target.rglob("*.parquet"))
    assert [path.parent.name for path in files] == ["region=__HIVE_DEFAULT_PARTITION__"]
    assert pq.read_table(files[0]).schema.names == ["id"]
    assert pq.read_table(files[0]).num_rows == 0


@pytest.mark.parametrize(
    ("existing", "appended"), [([], ["region"]), (["region"], []), (["region"], ["id"])]
)
def test_append_rejects_different_partitioning(existing: list[str], appended: list[str]) -> None:
    """Appends must use the partition columns of the existing data."""
    write_cache(TABLE, _table([1]), partition_by=existing)

    with pytest.raises(FabricWriteError, match="partitioned by"):
        write_cache(TABLE, _table([2]), "append", partition_by=appended, merge_schema=True)


def test_missing_partition_column_raises() -> None:
    """Partition columns must exist in the data."""
    with pytest.raises(FabricWriteError, match="missing"):
        write_cache(TABLE, _table([1]), partition_by=["missing"])


def test_write_error_keeps_existing_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed overwrite raises a cache error, removes the temp folder and keeps the data."""
    write_cache(TABLE, _table([1]))

    def fail(data: pa.Table, directory: Path, partition_by: list[str]) -> None:
        directory.mkdir(parents=True)
        raise OSError("disk full")

    monkeypatch.setattr(fabric_module, "_write_parquet", fail)

    with pytest.raises(FabricCacheError, match="disk full"):
        write_cache(TABLE, _table([2]), "overwrite")
    assert _read(TABLE.local_path) == [1]
    assert not TABLE.local_path.with_name(f"{TABLE.local_path.name}.tmp").exists()


def test_append_error_keeps_existing_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed append raises a cache error without removing the existing data."""
    write_cache(TABLE, _table([1]))
    monkeypatch.setattr(
        fabric_module, "_write_parquet", MagicMock(side_effect=pa.ArrowInvalid("bad data"))
    )

    with pytest.raises(FabricCacheError, match="bad data"):
        write_cache(TABLE, _table([2]), "append")
    assert _read(TABLE.local_path) == [1]
