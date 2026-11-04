"""Redirect Fabric Spark writes to the local cache."""

from pyspark.sql import DataFrame
from pyspark.sql.readwriter import DataFrameWriter as OriginalDataFrameWriter
from pyspark.sql.utils import to_str

from binaryrain_fabric_local_pyspark.exceptions import FabricWriteError
from binaryrain_fabric_local_pyspark.fabric import parse_fabric_path, parse_table_name, write_cache
from binaryrain_fabric_local_pyspark.schemas import FabricPath


# Formats that can be redirected, both are written as parquet. Fabric defaults to delta.
_REDIRECT_FORMATS = frozenset({"delta", "parquet"})


_DEFAULT_WRITE_FORMAT = "delta"


class DataFrameWriter(OriginalDataFrameWriter):
    """
    Redirect writes to Fabric lakehouses into the local cache instead of OneLake.

    ``saveAsTable`` always writes to the cache. ``save`` and ``parquet`` write to the cache
    when the path is an abfss URI or a default lakehouse relative path, and to the given
    local path otherwise. Delta and parquet writes are stored as parquet files, so no Delta
    Lake or Hadoop setup is needed and production data cannot be overwritten.
    """

    def __init__(self, df: DataFrame) -> None:
        """
        Create a writer for ``df`` that records its settings for redirected writes.

        :param DataFrame df:
            DataFrame to write.
        """
        super().__init__(df)
        self._save_mode: str | None = None
        self._source: str | None = None
        self._write_options: dict[str, str | None] = {}
        self._partition_columns: list[str] = []

    def mode(self, saveMode: str | None) -> "DataFrameWriter":
        """
        Set the save mode, see :meth:`pyspark.sql.DataFrameWriter.mode`.

        :param str | None saveMode:
            Spark save mode. None keeps the current mode.

        :returns DataFrameWriter:
            This writer.
        """
        if saveMode is not None:
            self._save_mode = saveMode
        super().mode(saveMode)
        return self

    def format(self, source: str) -> "DataFrameWriter":
        """
        Set the output format, see :meth:`pyspark.sql.DataFrameWriter.format`.

        :param str source:
            Output format such as ``delta`` or ``parquet``.

        :returns DataFrameWriter:
            This writer.
        """
        self._source = source
        super().format(source)
        return self

    def option(self, key: str, value: str | float | bool | None) -> "DataFrameWriter":
        """
        Add a write option, see :meth:`pyspark.sql.DataFrameWriter.option`.

        :param str key:
            Option name. Names are case insensitive.
        :param str | float | bool | None value:
            Option value.

        :returns DataFrameWriter:
            This writer.
        """
        self._write_options[key.lower()] = to_str(value)
        super().option(key, value)
        return self

    def options(self, **options: str | float | bool | None) -> "DataFrameWriter":
        """
        Add write options, see :meth:`pyspark.sql.DataFrameWriter.options`.

        :param options:
            Option names and values. Names are case insensitive.

        :returns DataFrameWriter:
            This writer.
        """
        self._write_options.update({key.lower(): to_str(value) for key, value in options.items()})
        super().options(**options)
        return self

    def partitionBy(self, *cols: str | list[str]) -> "DataFrameWriter":
        """
        Set partition columns, see :meth:`pyspark.sql.DataFrameWriter.partitionBy`.

        :param cols:
            Column names, or a single list of column names.

        :returns DataFrameWriter:
            This writer.
        """
        self._partition_columns = _column_list(cols[0] if len(cols) == 1 else list(cols))
        super().partitionBy(*cols)
        return self

    def save(
        self,
        path: str | None = None,
        format: str | None = None,
        mode: str | None = None,
        partitionBy: str | list[str] | None = None,
        **options: str | float | bool | None,
    ) -> None:
        """
        Save the DataFrame, redirecting Fabric lakehouse paths into the local cache.

        :param str | None path:
            abfss URI, default lakehouse relative path, or local path. Default is None.
        :param str | None format:
            Output format. Default is None, which uses the writer's format or ``delta``.
        :param str | None mode:
            Spark save mode. Default is None, which uses the writer's mode.
        :param str | list[str] | None partitionBy:
            Partition columns. Default is None, which uses the writer's columns.
        :param options:
            Additional write options.
        exception : FabricConfigurationError
            If a default lakehouse relative path is used without the required Fabric IDs.
        exception : FabricWriteError
            If a redirected write uses an unsupported format or conflicts with the cache.
        exception : FabricCacheError
            If the data cannot be written to the local cache.
        """
        fabric_path = parse_fabric_path(path) if path else None
        if fabric_path is None:
            super().save(path, format, mode, partitionBy, **options)
            return
        self._write_to_cache(fabric_path, format, mode, partitionBy, options)

    def saveAsTable(
        self,
        name: str,
        format: str | None = None,
        mode: str | None = None,
        partitionBy: str | list[str] | None = None,
        **options: str | float | bool | None,
    ) -> None:
        """
        Save the DataFrame as a lakehouse table in the local cache.

        The table is stored at the cache location of ``Tables/<schema>/<table>``. See
        ``parse_table_name`` for supported names. Read it back with ``spark.table``.

        :param str name:
            Table name such as ``customers``, ``dbo.customers`` or ``lh_gold.dbo.customers``.
        :param str | None format:
            Output format. Default is None, which uses the writer's format or ``delta``.
        :param str | None mode:
            Spark save mode. Default is None, which uses the writer's mode.
        :param str | list[str] | None partitionBy:
            Partition columns. Default is None, which uses the writer's columns.
        :param options:
            Additional write options.
        exception : ValueError
            If the table name is invalid.
        exception : FabricConfigurationError
            If the name relies on default Fabric IDs that are unset.
        exception : FabricWriteError
            If the format is unsupported or the write conflicts with the cache.
        exception : FabricCacheError
            If the data cannot be written to the local cache.
        """
        self._write_to_cache(parse_table_name(name), format, mode, partitionBy, options)

    def parquet(
        self,
        path: str,
        mode: str | None = None,
        partitionBy: str | list[str] | None = None,
        compression: str | None = None,
    ) -> None:
        """
        Save the DataFrame as parquet, redirecting Fabric lakehouse paths into the cache.

        :param str path:
            abfss URI, default lakehouse relative path, or local path.
        :param str | None mode:
            Spark save mode. Default is None, which uses the writer's mode.
        :param str | list[str] | None partitionBy:
            Partition columns. Default is None, which uses the writer's columns.
        :param str | None compression:
            Compression codec, ignored for redirected writes. Default is None.
        exception : FabricConfigurationError
            If a default lakehouse relative path is used without the required Fabric IDs.
        exception : FabricWriteError
            If a redirected write conflicts with the cache.
        exception : FabricCacheError
            If the data cannot be written to the local cache.
        """
        fabric_path = parse_fabric_path(path)
        if fabric_path is None:
            super().parquet(path, mode, partitionBy, compression)
            return
        self._write_to_cache(fabric_path, "parquet", mode, partitionBy, {})

    def _write_to_cache(
        self,
        fabric_path: FabricPath,
        source: str | None,
        mode: str | None,
        partition_by: str | list[str] | None,
        options: dict[str, str | float | bool | None],
    ) -> None:
        """
        Write the DataFrame to the cache location of a lakehouse path.

        Arguments override the settings made with ``mode``, ``format``, ``option`` and
        ``partitionBy``. The data is collected to the driver as an Arrow table.

        :param FabricPath fabric_path:
            Lakehouse location that would be written in Fabric.
        :param str | None source:
            Output format, or None to use the writer's format.
        :param str | None mode:
            Spark save mode, or None to use the writer's mode.
        :param str | list[str] | None partition_by:
            Partition columns, or None to use the writer's columns.
        :param dict[str, str | float | bool | None] options:
            Write options passed to the write call.
        exception : FabricWriteError
            If the format is unsupported or the write conflicts with the cache.
        exception : FabricCacheError
            If the data cannot be written to the local cache.
        """
        source = (source or self._source or _DEFAULT_WRITE_FORMAT).lower()
        if source not in _REDIRECT_FORMATS:
            raise FabricWriteError(
                f"Cannot write format {source!r} to {fabric_path.uri} locally. Only "
                f"{' and '.join(sorted(_REDIRECT_FORMATS))} writes are redirected to the cache."
            )
        write_options = {
            **self._write_options,
            **{key.lower(): to_str(value) for key, value in options.items()},
        }
        columns = _column_list(partition_by) if partition_by else self._partition_columns
        write_cache(
            fabric_path,
            self._df.toArrow(),
            mode or self._save_mode,
            columns,
            merge_schema=write_options.get("mergeschema") == "true",
        )


def _column_list(columns: str | list[str] | tuple[str, ...]) -> list[str]:
    """
    Normalize a column name or a sequence of column names to a list.

    :param str | list[str] | tuple[str, ...] columns:
        A single column name or a sequence of column names.

    :returns list[str]:
        The column names as a list.
    """
    return [columns] if isinstance(columns, str) else list(columns)
