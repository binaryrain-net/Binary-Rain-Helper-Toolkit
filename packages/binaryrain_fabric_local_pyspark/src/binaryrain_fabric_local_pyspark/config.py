"""Configuration constants and environment variables used by the package.

``FABRIC_CACHE_DIR`` overrides the cache root. When unset, cached data goes under
``.cache/fabric`` in the current working directory, which is normally the project root.
``spark.log`` and the SQL cache sit next to it in ``.cache``. Set ``FABRIC_CACHE_REFRESH=1``
to refresh cached data. ``FABRIC_WORKSPACE_ID`` and ``FABRIC_LAKEHOUSE_ID`` identify
the default lakehouse for relative paths. ``FABRIC_SERVER`` and ``FABRIC_DEFAULT_DATABASE``
configure the SQL endpoint, and ``FABRIC_SPARK_DEBUG=1`` enables verbose Spark output.
"""

import os
from pathlib import Path

CACHE_DIR = Path(os.getenv("FABRIC_CACHE_DIR", Path.cwd() / ".cache" / "fabric"))

LAKEHOUSE_AREAS = ("Files", "Tables")

MIN_ABFSS_SEGMENTS = 3  # <lakehouse>/<area>/<relative>
MIN_RELATIVE_SEGMENTS = 2  # <area>/<relative>

ONELAKE_HOST = "onelake.dfs.fabric.microsoft.com"
# Schema used for table names without a schema, matching schema-enabled lakehouses.
DEFAULT_TABLE_SCHEMA = "dbo"
# Hive folder value for null partition values, used by Spark and pyarrow.
HIVE_DEFAULT_PARTITION = "__HIVE_DEFAULT_PARTITION__"
# Spark save modes that fail when the target already exists.
ERROR_SAVE_MODES = ("error", "errorifexists", "default")
SAVE_MODES = ("append", "overwrite", "ignore", *ERROR_SAVE_MODES)
STORAGE_SCOPE = "https://storage.azure.com/.default"
FABRIC_TOKEN_SCOPE = "https://database.windows.net/.default"
# SQL_COPT_SS_ACCESS_TOKEN connection attribute for passing an AAD access token
# to the ODBC Driver for SQL Server.
SQL_COPT_SS_ACCESS_TOKEN = 1256
