# binaryrain-fabric-local-pyspark

Run Microsoft Fabric PySpark code on your own machine without paying for capacity on every run.

The package gives you a local Spark session. The first time your code reads a lakehouse table, a lakehouse file, or a Fabric SQL endpoint query, the data is downloaded to a cache folder in your project. Every later run reads from that folder and does not touch Fabric.

## Installation

```bash
pip install binaryrain-fabric-local-pyspark
```

Requirements:

- Python 3.12 or newer.
- Java 17 or newer, with `JAVA_HOME` set. PySpark needs it to start.
- ODBC Driver 18 for SQL Server, only if you use `spark.sql(...)` against the Fabric SQL endpoint.
- An Azure sign-in that can read the lakehouse, for example `az login`. The package uses `DefaultAzureCredential`.

## Quick start

Set the environment variables before you import the package. The cache location is read at import time.

```bash
FABRIC_WORKSPACE_ID=<workspace id or name>
FABRIC_LAKEHOUSE_ID=<lakehouse id or name>
FABRIC_SERVER=<sql endpoint host>.datawarehouse.fabric.microsoft.com
FABRIC_DEFAULT_DATABASE=<lakehouse name>
```

```python
from binaryrain_fabric_local_pyspark.spark import get_local_spark

spark = get_local_spark()

# Delta table in the default lakehouse, cached as parquet.
customers = spark.read.parquet("Tables/dbo/kunden")

# File or folder under Files, cached as is.
orders = spark.read.parquet("Files/onPrem/mssql/Orders.parquet")

# Full abfss URI, any workspace or lakehouse you can read.
other = spark.read.parquet(
    "abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>/Tables/dbo/kunden"
)

# T-SQL against the SQL endpoint. The result is cached by query text.
top = spark.sql("SELECT TOP 100 * FROM dbo.kunden")
```

`get_local_spark()` returns a normal `SparkSession` with two changes. `spark.read.parquet(...)` resolves Fabric paths through the cache, and `spark.sql(...)` runs your query on the Fabric SQL endpoint when you pass no parameters.

The Fabric SQL endpoint speaks T-SQL, not Spark SQL. If you pass `args` or keyword parameters, `spark.sql` skips the endpoint and runs on local Spark instead.

Paths that are not Fabric paths, such as `C:\data\file.parquet`, are passed through unchanged.

## What gets cached

| Source | Example input | How it is stored |
| --- | --- | --- |
| Delta table | `Tables/dbo/kunden` | Current table version written as parquet files |
| File or folder | `Files/raw/orders.parquet` | Downloaded byte for byte |
| SQL query | `spark.sql("SELECT ...")` | One parquet file, named by a SHA-256 hash of server and query text |

Default layout:

```text
<project>/.cache/
    fabric/<workspace>/<lakehouse>/<Tables|Files>/<relative path>
    sql/<hash>.parquet
    spark.log
```

A cache hit never checks Fabric. If the remote data changes, refresh the cache yourself (see below).

Downloads go to a `.tmp` path first and are moved into place when they finish, so a failed download does not leave a half-written cache entry. Add `.cache/` to your project's `.gitignore`.

## Configuration

All settings are environment variables.

| Variable | Purpose | Default |
| --- | --- | --- |
| `FABRIC_WORKSPACE_ID` | Workspace for relative paths like `Tables/...` | none, required for relative paths |
| `FABRIC_LAKEHOUSE_ID` | Lakehouse for relative paths like `Tables/...` | none, required for relative paths |
| `FABRIC_SERVER` | SQL endpoint host for `spark.sql` | none, required for `spark.sql` |
| `FABRIC_DEFAULT_DATABASE` | Database on the SQL endpoint | `default` |
| `FABRIC_CACHE_DIR` | Root folder for table and file cache | `<cwd>/.cache/fabric` |
| `FABRIC_CACHE_REFRESH` | Set to `1` to download again instead of reading the cache | unset |
| `FABRIC_SPARK_DEBUG` | Set to `1` for verbose Spark output | unset |

The default cache folder is relative to the directory you start Python from. Run your script from the project root, or set `FABRIC_CACHE_DIR` to an absolute path. `spark.log` and the `sql` cache sit one level above `FABRIC_CACHE_DIR`.

The package does not load `.env` files for you. Load them first, for example with `python-dotenv`, and do it before importing the package.

## Refreshing the cache

Set `FABRIC_CACHE_REFRESH=1` to refresh everything for one run. To refresh a single table or file, call the helpers directly:

```python
from binaryrain_fabric_local_pyspark.fabric import cache_file, cache_table, resolve_local_path

cache_table("Tables/dbo/kunden", refresh=True)
cache_file("Files/raw/orders.parquet", refresh=True)

# Returns a local path for Fabric inputs, the same string for anything else.
local_path = resolve_local_path("Tables/dbo/kunden")
```

You can also delete the folder under `.cache`.

## Errors

Every error the package raises on purpose derives from `FabricLocalSparkError` in `binaryrain_fabric_local_pyspark.exceptions`.

| Exception | Raised when |
| --- | --- |
| `FabricConfigurationError` | A required environment variable is missing, or a path is in the wrong area, such as a `Files/` path given to `cache_table`. Also a `ValueError`. |
| `FabricAuthenticationError` | No Azure token is available, or Fabric rejects the credentials. |
| `FabricDownloadError` | Reading a table or file from OneLake fails. |
| `FabricCacheError` | The cache folder or SQL result file cannot be written. |
| `FabricQueryError` | The SQL endpoint connection or query fails, or the engine cannot be created. |
| `SparkLaunchError` | The Spark JVM does not start. Usually Java is missing or `JAVA_HOME` is wrong. |

A path that does not exist in the lakehouse raises the built-in `FileNotFoundError`. A malformed abfss URI raises `ValueError`.

```python
from binaryrain_fabric_local_pyspark.exceptions import FabricAuthenticationError

try:
    df = spark.read.parquet("Tables/dbo/kunden")
except FabricAuthenticationError:
    print("Run 'az login' and try again.")
```

## Modules

| Module | Contents |
| --- | --- |
| `spark` | `get_local_spark`, plus the `SparkSession` and `DataFrameReader` subclasses |
| `fabric` | `parse_fabric_path`, `resolve_local_path`, `cache_table`, `cache_file` |
| `sql` | `get_fabric_engine`, `get_fabric_table_from_lakehouse` |
| `auth` | `get_token` |
| `schemas` | `FabricPath` |
| `exceptions` | The exception classes above |
| `config` | Environment variables and constants |

## Limits

- Tables are cached as plain parquet. Read them with `spark.read.parquet`, not `format("delta")`.
- Only `spark.read.parquet` is wired to the cache. Other readers such as `csv` or `json` get no Fabric path resolution.
- The cache has no expiry and no size limit.
- The package only reads from Fabric. It never writes back to OneLake or the SQL endpoint.

## License

See [LICENSE](LICENSE).
