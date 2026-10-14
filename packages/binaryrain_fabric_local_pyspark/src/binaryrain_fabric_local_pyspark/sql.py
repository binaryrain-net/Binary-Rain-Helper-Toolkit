"""Connect to a Fabric lakehouse SQL endpoint using Azure token authentication."""

import os
import struct
from functools import cache
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from binaryrain_fabric_local_pyspark.auth import get_token
from binaryrain_fabric_local_pyspark.config import FABRIC_TOKEN_SCOPE, SQL_COPT_SS_ACCESS_TOKEN
from binaryrain_fabric_local_pyspark.exceptions import FabricConfigurationError, FabricQueryError


@cache
def get_fabric_engine(
    server: str | None = None,
    database: str | None = None,
) -> Engine:
    """Create a cached SQLAlchemy engine for a Fabric lakehouse SQL endpoint.

    Each new physical connection receives a fresh Azure Active Directory access token.
    ``server`` and ``database`` override ``FABRIC_SERVER`` and ``FABRIC_DEFAULT_DATABASE``.
    Creating the engine does not open a connection, so connection and authentication
    errors surface on the first query.

    :param str | None server:
        Fabric SQL endpoint host. Defaults to ``FABRIC_SERVER``.
    :param str | None database:
        Lakehouse database name. Defaults to ``FABRIC_DEFAULT_DATABASE``, then ``default``.

    :returns Engine:
        SQLAlchemy engine configured for ODBC Driver 18 and Azure token authentication.
    exception : FabricConfigurationError
        If neither ``server`` nor ``FABRIC_SERVER`` provides a server name.
    exception : FabricQueryError
        If SQLAlchemy cannot create the engine, for example because the connection
        string is invalid.
    """
    fabric_server = server or os.getenv("FABRIC_SERVER")
    database = database or os.getenv("FABRIC_DEFAULT_DATABASE", "default")
    if not fabric_server:
        raise FabricConfigurationError(
            "No Fabric SQL endpoint configured. Set the FABRIC_SERVER environment variable "
            "or pass the server argument."
        )

    connection_string = (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server={fabric_server};"
        f"Database={database};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
    )
    url = f"mssql+pyodbc:///?odbc_connect={quote_plus(connection_string)}"
    try:
        engine = create_engine(url)
    except SQLAlchemyError as exc:
        raise FabricQueryError(
            f"Could not create the SQL engine for {fabric_server}: {exc}"
        ) from exc

    @event.listens_for(engine, "do_connect")
    def provide_token(dialect, conn_rec, cargs, cparams):
        """Set a fresh Azure access token on the ODBC connection parameters."""
        token = get_token(FABRIC_TOKEN_SCOPE).encode("utf-16-le")
        token_struct = struct.pack(f"<I{len(token)}s", len(token), token)
        cparams["attrs_before"] = {SQL_COPT_SS_ACCESS_TOKEN: token_struct}

    return engine


def get_fabric_table_from_lakehouse(
    engine: Engine,
    query: str,
) -> pd.DataFrame:
    """Execute a SQL query against Fabric and return its rows as a DataFrame.

    :param Engine engine:
        SQLAlchemy engine connected to the Fabric lakehouse endpoint.
    :param str query:
        SQL query to execute.

    :returns pd.DataFrame:
        Rows returned by the query, with columns named from the result set.
    exception : FabricAuthenticationError
        If an Azure access token cannot be acquired when the connection opens.
    exception : FabricQueryError
        If the connection or the query fails. The message includes the driver error.
        A missing "ODBC Driver 18 for SQL Server" also surfaces here.
    """
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except (SQLAlchemyError, pd.errors.DatabaseError) as exc:
        raise FabricQueryError(
            f"Fabric SQL query failed: {exc}. Check the query, the server name, and that "
            "ODBC Driver 18 for SQL Server is installed."
        ) from exc
