"""Connect to a Fabric lakehouse SQL endpoint using Azure token authentication."""

import os
import struct
from functools import cache
from urllib.parse import quote_plus

import pandas as pd
from azure.identity import DefaultAzureCredential
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from binaryrain_helper_local_pyspark.config import FABRIC_TOKEN_SCOPE, SQL_COPT_SS_ACCESS_TOKEN


@cache
def get_fabric_engine(
    server: str | None = None,
    database: str | None = None,
) -> Engine:
    """Create a cached SQLAlchemy engine for a Fabric lakehouse SQL endpoint.

    Each new physical connection receives a fresh Azure Active Directory access token.
    ``server`` and ``database`` override ``FABRIC_SERVER`` and ``FABRIC_DATABASE``.

    :param str | None server:
        Fabric SQL endpoint host. Defaults to ``FABRIC_SERVER``.
    :param str | None database:
        Lakehouse database name. Defaults to ``FABRIC_DATABASE``, then ``default``.

    :returns Engine:
        SQLAlchemy engine configured for ODBC Driver 18 and Azure token authentication.
    exception : ValueError
        If neither ``server`` nor ``FABRIC_SERVER`` provides a server name.
    """

    fabric_server = server or os.getenv("FABRIC_SERVER")
    database = database or os.getenv("FABRIC_DATABASE", "default")
    if not fabric_server:
        raise ValueError("FABRIC_SERVER environment variables must be set.")

    connection_string = (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server={fabric_server};"
        f"Database={database or 'default'};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
    )
    url = f"mssql+pyodbc:///?odbc_connect={quote_plus(connection_string)}"
    engine = create_engine(url)

    credential = DefaultAzureCredential()

    @event.listens_for(engine, "do_connect")
    def provide_token(dialect, conn_rec, cargs, cparams):
        """Set the Azure access token on the ODBC connection parameters."""
        token = credential.get_token(FABRIC_TOKEN_SCOPE).token.encode("utf-16-le")
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
    """
    result = pd.DataFrame()
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)
    return result
