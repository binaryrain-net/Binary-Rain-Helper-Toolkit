"""Tests for Fabric path parsing and the local OneLake cache."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from azure.core.exceptions import AzureError, ClientAuthenticationError, ResourceNotFoundError
from deltalake.exceptions import DeltaError, TableNotFoundError

from binaryrain_fabric_local_pyspark import fabric as fabric_module
from binaryrain_fabric_local_pyspark import schemas as schemas_module
from binaryrain_fabric_local_pyspark.config import ONELAKE_HOST
from binaryrain_fabric_local_pyspark.exceptions import (
    FabricAuthenticationError,
    FabricCacheError,
    FabricConfigurationError,
    FabricDownloadError,
)
from binaryrain_fabric_local_pyspark.fabric import (
    _commit,
    _remove,
    cache_file,
    cache_table,
    parse_fabric_path,
    resolve_local_path,
)
from binaryrain_fabric_local_pyspark.schemas import FabricPath

FILE_ROOT = "lh/Files/raw/kunden.parquet"


@pytest.fixture(autouse=True)
def fabric_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the default lakehouse and disable forced cache refreshes."""
    monkeypatch.setenv("FABRIC_WORKSPACE_ID", "ws")
    monkeypatch.setenv("FABRIC_LAKEHOUSE_ID", "lh")
    monkeypatch.delenv("FABRIC_CACHE_REFRESH", raising=False)


@pytest.fixture
def cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the local cache at a temporary folder."""
    directory = tmp_path / "cache"
    monkeypatch.setattr(schemas_module, "CACHE_DIR", directory)
    return directory


@pytest.fixture
def file_system(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the OneLake client with a mock file system client."""
    service = MagicMock()
    monkeypatch.setattr(fabric_module, "DataLakeServiceClient", MagicMock(return_value=service))
    monkeypatch.setattr(fabric_module, "_credential", MagicMock())
    return service.get_file_system_client.return_value


@pytest.fixture
def delta_table(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace DeltaTable with a mock that returns a small in-memory dataset."""
    mock = MagicMock()
    mock.return_value.to_pyarrow_dataset.return_value = ds.dataset(pa.table({"id": [1, 2]}))
    monkeypatch.setattr(fabric_module, "DeltaTable", mock)
    monkeypatch.setattr(fabric_module, "_storage_token", MagicMock(return_value="token"))
    return mock


def _serve_files(file_system: MagicMock, files: dict[str, bytes], is_folder: bool) -> None:
    """
    Let the mock file system serve ``files`` below ``FILE_ROOT``.

    :param MagicMock file_system:
        Mock returned by the ``file_system`` fixture.
    :param dict[str, bytes] files:
        Remote file paths mapped to their content.
    :param bool is_folder:
        Whether ``FILE_ROOT`` is a folder rather than a single file.
    """

    def get_file_client(path: str) -> MagicMock:
        client = MagicMock()
        client.get_file_properties.return_value.metadata = {
            "hdi_isfolder": "true" if is_folder else "false"
        }
        client.download_file.return_value.readinto.side_effect = lambda handle: handle.write(
            files[path]
        )
        return client

    file_system.get_file_client.side_effect = get_file_client
    file_system.get_paths.return_value = [
        SimpleNamespace(name=f"{FILE_ROOT}/_delta", is_directory=True),
        *(SimpleNamespace(name=name, is_directory=False) for name in files),
    ]


# parse_fabric_path


def test_parse_abfss_uri() -> None:
    """An abfss URI is split into workspace, lakehouse, area, path and host."""
    result = parse_fabric_path("abfss://ws@host.example/lh.Lakehouse/Files/raw/a.parquet")

    assert result == FabricPath("ws", "lh.Lakehouse", "Files", "raw/a.parquet", "host.example")


def test_parse_abfss_uri_is_case_insensitive() -> None:
    """The abfss scheme is recognized regardless of case."""
    result = parse_fabric_path("ABFSS://ws@host/lh/Tables/dbo/kunden")

    assert result == FabricPath("ws", "lh", "Tables", "dbo/kunden", "host")


@pytest.mark.parametrize(
    "path",
    [
        "abfss://host/lh/Files/a.parquet",
        "abfss://@host/lh/Files/a.parquet",
        "abfss://ws@host/lh/Files",
        "abfss://ws@host/lh/Other/a.parquet",
    ],
)
def test_parse_malformed_abfss_uri_raises(path: str) -> None:
    """Malformed abfss URIs raise ValueError."""
    with pytest.raises(ValueError, match="Expected abfss://"):
        parse_fabric_path(path)


@pytest.mark.parametrize(
    "path",
    ["Files/raw/a.parquet", "/Files/raw/a.parquet", "Files\\raw\\a.parquet"],
)
def test_parse_relative_path_uses_default_lakehouse(path: str) -> None:
    """Relative lakehouse paths resolve against the default lakehouse."""
    assert parse_fabric_path(path) == FabricPath("ws", "lh", "Files", "raw/a.parquet")


@pytest.mark.parametrize("path", ["data/a.parquet", "Files", "C:/data/a.parquet", "files/a"])
def test_parse_non_fabric_path_returns_none(path: str) -> None:
    """Paths outside a lakehouse area are not Fabric paths."""
    assert parse_fabric_path(path) is None


@pytest.mark.parametrize("variable", ["FABRIC_WORKSPACE_ID", "FABRIC_LAKEHOUSE_ID"])
def test_parse_relative_path_without_ids_raises(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    """Relative paths require both default lakehouse IDs."""
    monkeypatch.delenv(variable)

    with pytest.raises(FabricConfigurationError, match=variable):
        parse_fabric_path("Tables/dbo/kunden")


# resolve_local_path


@pytest.fixture
def cache_mocks(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace cache_table and cache_file with mocks."""
    mocks = SimpleNamespace(
        table=MagicMock(return_value=Path("table")), file=MagicMock(return_value=Path("file"))
    )
    monkeypatch.setattr(fabric_module, "cache_table", mocks.table)
    monkeypatch.setattr(fabric_module, "cache_file", mocks.file)
    return mocks


def test_resolve_non_fabric_path_is_unchanged(cache_mocks: SimpleNamespace) -> None:
    """A local path is returned as is without touching the cache."""
    assert resolve_local_path("data/a.parquet") == "data/a.parquet"
    cache_mocks.table.assert_not_called()
    cache_mocks.file.assert_not_called()


def test_resolve_tables_path_uses_cache_table(cache_mocks: SimpleNamespace) -> None:
    """Tables paths are cached as Delta tables."""
    assert resolve_local_path("Tables/dbo/kunden") == "table"
    cache_mocks.table.assert_called_once_with(FabricPath("ws", "lh", "Tables", "dbo/kunden"), False)


def test_resolve_files_path_uses_cache_file(cache_mocks: SimpleNamespace) -> None:
    """Files paths are cached as files or folders."""
    assert resolve_local_path("Files/raw/a.parquet", refresh=True) == "file"
    cache_mocks.file.assert_called_once_with(FabricPath("ws", "lh", "Files", "raw/a.parquet"), True)


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("0", False)])
def test_resolve_reads_refresh_from_environment(
    monkeypatch: pytest.MonkeyPatch, cache_mocks: SimpleNamespace, value: str, expected: bool
) -> None:
    """FABRIC_CACHE_REFRESH controls refreshing when refresh is None."""
    monkeypatch.setenv("FABRIC_CACHE_REFRESH", value)

    resolve_local_path("Files/raw/a.parquet")

    assert cache_mocks.file.call_args.args[1] is expected


# cache_table


def test_cache_table_downloads_delta_table(cache_dir: Path, delta_table: MagicMock) -> None:
    """A cache miss writes the Delta table as parquet into the cache."""
    target = cache_table("Tables/dbo/kunden")

    assert target == cache_dir / "ws" / "lh" / "Tables" / "dbo" / "kunden"
    assert ds.dataset(target).to_table().column("id").to_pylist() == [1, 2]
    assert not target.with_name("kunden.tmp").exists()
    delta_table.assert_called_once_with(
        f"abfss://ws@{ONELAKE_HOST}/lh/Tables/dbo/kunden",
        storage_options={"bearer_token": "token", "use_fabric_endpoint": "true"},
    )


def test_cache_table_hit_skips_download(cache_dir: Path, delta_table: MagicMock) -> None:
    """An existing cache entry is returned without reading the remote table."""
    target = cache_dir / "ws" / "lh" / "Tables" / "dbo" / "kunden"
    target.mkdir(parents=True)

    assert cache_table("Tables/dbo/kunden") == target
    delta_table.assert_not_called()


def test_cache_table_refresh_replaces_cache(cache_dir: Path, delta_table: MagicMock) -> None:
    """refresh=True downloads again and replaces the old cache entry."""
    target = cache_dir / "ws" / "lh" / "Tables" / "dbo" / "kunden"
    target.mkdir(parents=True)
    (target / "stale.parquet").write_bytes(b"stale")

    cache_table("Tables/dbo/kunden", refresh=True)

    assert not (target / "stale.parquet").exists()
    delta_table.assert_called_once()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TableNotFoundError("missing"), FileNotFoundError),
        (DeltaError("denied"), FabricDownloadError),
        (pa.ArrowInvalid("broken"), FabricDownloadError),
    ],
)
def test_cache_table_maps_read_errors(
    cache_dir: Path, delta_table: MagicMock, error: Exception, expected: type[Exception]
) -> None:
    """Delta and Arrow errors are mapped to package errors and leave no cache behind."""
    delta_table.side_effect = error
    target = cache_dir / "ws" / "lh" / "Tables" / "dbo" / "kunden"

    with pytest.raises(expected):
        cache_table("Tables/dbo/kunden")

    assert not target.exists()
    assert not target.with_name("kunden.tmp").exists()


def test_cache_table_write_error_raises_cache_error(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path, delta_table: MagicMock
) -> None:
    """A failure while writing the parquet files raises FabricCacheError."""
    monkeypatch.setattr(fabric_module.ds, "write_dataset", MagicMock(side_effect=OSError("full")))

    with pytest.raises(FabricCacheError, match="full"):
        cache_table("Tables/dbo/kunden")


@pytest.mark.parametrize("path", ["Files/raw/a.parquet", "data/a.parquet"])
def test_cache_table_rejects_other_paths(path: str) -> None:
    """cache_table only accepts lakehouse Tables paths."""
    with pytest.raises(FabricConfigurationError, match="Tables"):
        cache_table(path)


# cache_file


def test_cache_file_downloads_single_file(cache_dir: Path, file_system: MagicMock) -> None:
    """A single remote file is downloaded to the cache."""
    _serve_files(file_system, {FILE_ROOT: b"content"}, is_folder=False)

    target = cache_file("Files/raw/kunden.parquet")

    assert target == cache_dir / "ws" / "lh" / "Files" / "raw" / "kunden.parquet"
    assert target.read_bytes() == b"content"
    file_system.get_paths.assert_not_called()


def test_cache_file_downloads_folder(cache_dir: Path, file_system: MagicMock) -> None:
    """A remote folder is downloaded file by file, keeping its layout."""
    files = {
        f"{FILE_ROOT}/part-0.parquet": b"zero",
        f"{FILE_ROOT}/nested/part-1.parquet": b"one",
    }
    _serve_files(file_system, files, is_folder=True)

    target = cache_file(f"abfss://ws@{ONELAKE_HOST}/{FILE_ROOT}")

    assert (target / "part-0.parquet").read_bytes() == b"zero"
    assert (target / "nested" / "part-1.parquet").read_bytes() == b"one"
    assert not (target / "_delta").exists()


def test_cache_file_hit_skips_download(cache_dir: Path, file_system: MagicMock) -> None:
    """An existing cache entry is returned without contacting OneLake."""
    target = cache_dir / "ws" / "lh" / "Files" / "raw" / "kunden.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"cached")

    assert cache_file("Files/raw/kunden.parquet") == target
    fabric_module.DataLakeServiceClient.assert_not_called()


def test_cache_file_refresh_replaces_cache(cache_dir: Path, file_system: MagicMock) -> None:
    """refresh=True downloads again and overwrites the cached file."""
    target = cache_dir / "ws" / "lh" / "Files" / "raw" / "kunden.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"stale")
    _serve_files(file_system, {FILE_ROOT: b"fresh"}, is_folder=False)

    cache_file("Files/raw/kunden.parquet", refresh=True)

    assert target.read_bytes() == b"fresh"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ResourceNotFoundError("missing"), FileNotFoundError),
        (ClientAuthenticationError("denied"), FabricAuthenticationError),
        (AzureError("timeout"), FabricDownloadError),
        (OSError("full"), FabricCacheError),
    ],
)
def test_cache_file_maps_errors(
    cache_dir: Path, file_system: MagicMock, error: Exception, expected: type[Exception]
) -> None:
    """Download errors are mapped to package errors and leave no temp file behind."""
    _serve_files(file_system, {FILE_ROOT: b"content"}, is_folder=False)
    file_system.get_file_client.side_effect = None
    file_system.get_file_client.return_value.get_file_properties.return_value.metadata = {}
    file_system.get_file_client.return_value.download_file.side_effect = error

    with pytest.raises(expected):
        cache_file("Files/raw/kunden.parquet")

    assert not (cache_dir / "ws" / "lh" / "Files" / "raw" / "kunden.parquet.tmp").exists()


def test_cache_file_not_found_includes_error_code(cache_dir: Path, file_system: MagicMock) -> None:
    """The Storage error code is part of the FileNotFoundError message."""
    error = ResourceNotFoundError("missing")
    error.error_code = "PathNotFound"
    file_system.get_file_client.return_value.get_file_properties.side_effect = error

    with pytest.raises(FileNotFoundError, match="PathNotFound"):
        cache_file("Files/raw/kunden.parquet")


def test_cache_file_rejects_tables_path() -> None:
    """cache_file only accepts lakehouse Files paths."""
    with pytest.raises(FabricConfigurationError, match="Files"):
        cache_file("Tables/dbo/kunden")


# _remove and _commit


def test_remove_deletes_file_and_folder(tmp_path: Path) -> None:
    """_remove deletes files and folders and ignores missing paths."""
    file = tmp_path / "file.txt"
    file.write_text("x")
    folder = tmp_path / "folder"
    (folder / "child").mkdir(parents=True)

    _remove(file)
    _remove(folder)
    _remove(tmp_path / "missing")

    assert not file.exists()
    assert not folder.exists()


def test_remove_raises_unless_quiet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Deletion errors are raised, or logged when quiet is True."""
    monkeypatch.setattr(fabric_module.shutil, "rmtree", MagicMock(side_effect=OSError("locked")))

    with pytest.raises(OSError, match="locked"):
        _remove(tmp_path)
    _remove(tmp_path, quiet=True)

    assert "Could not remove temporary cache path" in caplog.text


def test_commit_replaces_target(tmp_path: Path) -> None:
    """_commit moves the temp download into place and replaces the old target."""
    tmp = tmp_path / "data.tmp"
    tmp.write_text("new")
    target = tmp_path / "nested" / "data"
    target.parent.mkdir()
    target.write_text("old")

    _commit(tmp, target)

    assert target.read_text() == "new"
    assert not tmp.exists()


def test_commit_failure_raises_cache_error(tmp_path: Path) -> None:
    """_commit raises FabricCacheError and removes the temp download on failure."""
    tmp = tmp_path / "data.tmp"
    tmp.write_text("new")
    blocker = tmp_path / "blocker"
    blocker.write_text("not a folder")

    with pytest.raises(FabricCacheError, match="Could not move the download"):
        _commit(tmp, blocker / "data")

    assert not tmp.exists()
