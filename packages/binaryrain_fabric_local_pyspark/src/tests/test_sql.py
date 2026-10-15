"""Tests for the Fabric SQL endpoint engine and query helper."""

import struct
from collections.abc import Iterator
from unittest.mock import MagicMock
from urllib.parse import unquote_plus

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ArgumentError

from binaryrain_fabric_local_pyspark import sql as sql_module
from binaryrain_fabric_local_pyspark.config import FABRIC_TOKEN_SCOPE, SQL_COPT_SS_ACCESS_TOKEN
from binaryrain_fabric_local_pyspark.exceptions import FabricConfigurationError, FabricQueryError
from binaryrain_fabric_local_pyspark.sql import get_fabric_engine, get_fabric_table_from_lakehouse


@pytest.fixture(autouse=True)
def clear_engine_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset the cached engines and the SQL environment variables."""
    monkeypatch.delenv("FABRIC_SERVER", raising=False)
    monkeypatch.delenv("FABRIC_DEFAULT_DATABASE", raising=False)
    get_fabric_engine.cache_clear()
    yield
    get_fabric_engine.cache_clear()


def _odbc_connect(engine: Engine) -> str:
    """
    Return the decoded ODBC connection string of an engine.

    :param Engine engine:
        SQLAlchemy engine created by ``get_fabric_engine``.

    :returns str:
        The ``odbc_connect`` query value of the engine URL.
    """
    return unquote_plus(engine.url.query["odbc_connect"])


def test_engine_without_server_raises() -> None:
    """A missing server name raises FabricConfigurationError."""
    with pytest.raises(FabricConfigurationError, match="FABRIC_SERVER"):
        get_fabric_engine()


def test_engine_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server and database default to the environment variables."""
    monkeypatch.setenv("FABRIC_SERVER", "env.server")
    monkeypatch.setenv("FABRIC_DEFAULT_DATABASE", "env_db")

    connect = _odbc_connect(get_fabric_engine())

    assert "Server=env.server;" in connect
    assert "Database=env_db;" in connect
    assert "Driver={ODBC Driver 18 for SQL Server};" in connect
    assert "Encrypt=yes;" in connect


def test_engine_database_defaults_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without FABRIC_DEFAULT_DATABASE the database is called default."""
    monkeypatch.setenv("FABRIC_SERVER", "env.server")

    assert "Database=default;" in _odbc_connect(get_fabric_engine())


def test_engine_arguments_override_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit arguments take precedence over the environment."""
    monkeypatch.setenv("FABRIC_SERVER", "env.server")
    monkeypatch.setenv("FABRIC_DEFAULT_DATABASE", "env_db")

    connect = _odbc_connect(get_fabric_engine("arg.server", "arg_db"))

    assert "Server=arg.server;" in connect
    assert "Database=arg_db;" in connect


def test_engine_is_cached() -> None:
    """The same arguments return the same engine."""
    assert get_fabric_engine("server") is get_fabric_engine("server")
    assert get_fabric_engine("server") is not get_fabric_engine("other")


def test_engine_creation_error_raises_query_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """SQLAlchemy errors during engine creation raise FabricQueryError."""
    monkeypatch.setattr(sql_module, "create_engine", MagicMock(side_effect=ArgumentError("bad")))

    with pytest.raises(FabricQueryError, match="Could not create the SQL engine"):
        get_fabric_engine("server")


def test_connect_sets_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each new connection receives a fresh token in the ODBC attributes."""
    get_token = MagicMock(return_value="abc")
    monkeypatch.setattr(sql_module, "get_token", get_token)
    engine = get_fabric_engine("server")
    cparams: dict[str, object] = {}

    engine.dialect.dispatch.do_connect(engine.dialect, MagicMock(), [], cparams)

    token = "abc".encode("utf-16-le")
    expected = struct.pack(f"<I{len(token)}s", len(token), token)
    assert cparams["attrs_before"] == {SQL_COPT_SS_ACCESS_TOKEN: expected}
    get_token.assert_called_once_with(FABRIC_TOKEN_SCOPE)


@pytest.fixture
def sqlite_engine() -> Iterator[Engine]:
    """Provide an in-memory SQLite engine and close its connections afterwards."""
    engine = create_engine("sqlite://")
    yield engine
    engine.dispose()


def test_query_returns_dataframe(sqlite_engine: Engine) -> None:
    """Query results are returned as a pandas DataFrame."""
    result = get_fabric_table_from_lakehouse(sqlite_engine, "SELECT 1 AS id, 'a' AS name")

    pd.testing.assert_frame_equal(result, pd.DataFrame({"id": [1], "name": ["a"]}))


def test_query_error_raises_query_error(sqlite_engine: Engine) -> None:
    """Failing queries raise FabricQueryError with a hint."""
    with pytest.raises(FabricQueryError, match="ODBC Driver 18"):
        get_fabric_table_from_lakehouse(sqlite_engine, "SELECT * FROM missing_table")
