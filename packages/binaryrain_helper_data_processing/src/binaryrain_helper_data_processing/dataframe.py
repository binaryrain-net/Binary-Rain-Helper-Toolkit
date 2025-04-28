import io
from enum import Enum
import pandas as pd


class FileFormat(Enum):
    """
    The file formats supported for dataframe creation and conversion.
    """

    PARQUET = 1
    CSV = 2
    DICT = 3
    JSON = 4


def create_dataframe(
    file_contents: bytes | dict,
    file_format: FileFormat,
    file_format_options: dict | None = None,
) -> pd.DataFrame:
    """
    Create a dataframe from the file contents.

    :param bytes | dict file_contents:
        The contents of the file to be loaded.
    :param FileFormat file_format:
        The format of the file to be loaded. Currently supported: `csv` and `dict`, `parquet`, `json`.
    :param dict | None file_format_options:
        The options for the file format. Default is None.

    :returns pandas.DataFrame:
        The dataframe created from the file contents
    exception : ValueError
        If an error occurs during dataframe creation
    """
    try:
        match file_format:
            case FileFormat.CSV:
                if file_format_options is None:
                    dataframe = pd.read_csv(io.BytesIO(file_contents))
                else:
                    dataframe = pd.read_csv(
                        io.BytesIO(file_contents), **file_format_options
                    )

            case FileFormat.DICT:
                if file_format_options is None:
                    dataframe = pd.DataFrame.from_dict(file_contents)
                else:
                    dataframe = pd.DataFrame.from_dict(
                        file_contents, **file_format_options
                    )

            case FileFormat.PARQUET:
                if file_format_options is None:
                    dataframe = pd.read_parquet(
                        io.BytesIO(file_contents), engine="pyarrow"
                    )
                else:
                    dataframe = pd.read_parquet(
                        io.BytesIO(file_contents),
                        engine="pyarrow",
                        **file_format_options,
                    )

            case FileFormat.JSON:
                if file_format_options is None:
                    dataframe = pd.read_json(io.BytesIO(file_contents))
                else:
                    dataframe = pd.read_json(
                        io.BytesIO(file_contents), **file_format_options
                    )

            case _:
                raise TypeError(
                    f"Error creating dataframe. Unknown file format: {file_format}"
                )
    except Exception as exc:
        raise ValueError(f"Error creating dataframe. Exception: {exc}") from exc

    return dataframe


def from_dataframe_to_type(
    dataframe: pd.DataFrame,
    file_format: FileFormat,
    file_format_options: dict | None = None,
) -> bytes | str | dict:
    """
    Converts the dataframe to a specific file format.

    :param bytes | dict file_contents:
        The contents of the file to be loaded.
    :param FileFormat file_format:
        The format of the file to be loaded.
    :param dict | None file_format_options:
        The options for the file format. Default is None.

    :returns bytes:
        The file contents
    exception : ValueError
        If an error occurs during dataframe conversion
    """
    try:
        match file_format:
            case FileFormat.CSV:
                if file_format_options is None:
                    content = dataframe.to_csv(index=False)
                else:
                    content = dataframe.to_csv(index=False, **file_format_options)

            case FileFormat.DICT:
                if file_format_options is None:
                    content = dataframe.to_dict(orient="records")
                else:
                    content = dataframe.to_dict(**file_format_options)

            case FileFormat.PARQUET:
                if file_format_options is None:
                    content = dataframe.to_parquet(engine="pyarrow")
                else:
                    content = dataframe.to_parquet(
                        engine="pyarrow", **file_format_options
                    )

            case FileFormat.JSON:
                if file_format_options is None:
                    content = dataframe.to_json()
                else:
                    content = dataframe.to_json(**file_format_options)

            case _:
                raise TypeError(
                    f"Error converting dataframe. Unknown file format: {file_format}"
                )
    except Exception as exc:
        raise ValueError(
            f"Error converting dataframe. See logs for more details. Exception: {exc}"
        ) from exc

    return content


def combine_dataframes(
    df_one: pd.DataFrame | None,
    df_two: pd.DataFrame | None,
    sort: bool = False,
) -> pd.DataFrame:
    """
    Combine two dataframes.

    :param pd.DataFrame df_one:
        The first dataframe.
    :param pd.DataFrame df_two:
        The second dataframe.
    :param bool, optional sort:
        Sort the resulting dataframe. Default is False.

    :returns pandas.DataFrame df_merged:
        The merged dataframe.
    exception : ValueError
        If an error occurs during dataframe merging
    """
    if isinstance(df_one, pd.DataFrame) and isinstance(df_two, pd.DataFrame):
        try:
            df_merged = pd.concat([df_one, df_two], sort=sort)
        except Exception as exc:
            raise ValueError(f"Error merging dataframes. Exception: {exc}") from exc
    else:
        raise ValueError(
            f"No dataframe provided for df_one - got {type(df_one)} and/or df_two - got {type(df_two)}."
        )

    return df_merged


def convert_to_datetime(
    df: pd.DataFrame,
    date_formats: list[str] | None = ["%d.%m.%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"],
) -> pd.DataFrame:
    """
    Convert date columns in a dataframe to datetime format.
    Currently supports the following date formats:
    - "%d.%m.%Y"
    - "%Y-%m-%d"
    - "%Y-%m-%d %H:%M:%S"
    - "%Y-%m-%dT%H:%M:%S"

    :param pd.DataFrame df:
        The dataframe to convert date columns in.

    :returns pd.DataFrame df:
        The dataframe with date columns converted to datetime
    """
    for column in df.columns:
        # Only process object (string) columns
        if df[column].dtype == "object":
            # Try each date format. If conversion succeeds, break the loop
            for date_format in date_formats:
                try:
                    df[column] = pd.to_datetime(
                        df[column], format=date_format, errors="coerce"
                    )
                    break  # Exit the loop if conversion succeeds
                except (ValueError, TypeError):
                    continue
    return df


def format_datetime_columns(
    df: pd.DataFrame,
    datetime_columns: list[str],
    datetime_format: str,
    datetime_string_columns: list[str] = None,
) -> pd.DataFrame:
    """
    Format datetime columns in a dataframe to a specific format

    :param pd.DataFrame df:
        The dataframe to format datetime columns in.
    :param list[str] datetime_columns:
        The columns to format as datetime.
    :param str datetime_format:
        The format to convert the datetime columns to.
    :param list[str] datetime_string_columns:
        The columns to format as datetime strings. Optional. If not provided, the same columns as datetime_columns will be used.

    :returns pd.DataFrame df:
        The dataframe with datetime columns formatted to the specific format
    """
    if datetime_string_columns is None:
        datetime_string_columns = datetime_columns
    if len(datetime_columns) != len(datetime_string_columns):
        raise ValueError(
            "The number of datetime columns and datetime string columns must be equal."
        )
    for i in range(len(datetime_columns)):
        df[datetime_columns[i]] = pd.to_datetime(
            df[datetime_string_columns[i]]
        ).strftime(datetime_format)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a dataframe by removing rows with missing values.

    :param pd.DataFrame df:
        The dataframe to clean.

    :returns pd.DataFrame df:
        The cleaned dataframe
    """
    df = df.replace("nan", pd.NA)
    df = df.replace("", pd.NA)
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def remove_empty_values(df: pd.DataFrame, filter_column: str) -> pd.DataFrame:
    """
    Remove empty values from a dataframe.

    :param pd.DataFrame df:
        The dataframe to remove empty values from.
    :param str filter_column:
        The column to filter empty values on.

    :returns pd.DataFrame df:
        The dataframe with empty values removed.
    """
    return df[df[filter_column].notnull() & df[filter_column] != ""].reset_index(
        drop=True
    )


def format_numeric_values(
    df: pd.DataFrame,
    columns: list[str],
    swap_separators: bool = False,
    decimal_separator: str = ".",
    thousands_separator: str = ",",
    old_decimal_separator: str = ",",
    old_thousands_separator: str = ".",
    temp_separator: str = "|",
) -> pd.DataFrame:
    """
    Format numeric values in a dataframe. Additionally it swaps the decimal and thousands separators. This is useful when the data is read from a file with a different locale.

    :param pd.DataFrame df:
        The dataframe to format numeric values in.
    :param list[str] columns:
        The columns to format as numeric values.
    :param bool swap_separators:
        Swap the decimal and thousands separators. Default is False.
    :param str decimal_separator:
            The decimal separator to use. Default is `.`.
    :param str thousands_separator:
                The thousands separator to use. Default is `,`.
    :param str old_decimal_separator:
        The old decimal separator to replace. Default is `,`.
    :param str old_thousands_separator:
        The old thousands separator to replace. Default is `.`.
    :param str temp_separator:
        The temporary separator to use. Default is `|`.

    :returns pd.DataFrame df:
        The dataframe with numeric values formatted
    """
    for column in columns:
        if swap_separators:
            df[column] = (
                df[column]
                .str.replace(old_thousands_separator, temp_separator)
                .str.replace(old_decimal_separator, decimal_separator)
                .str.replace(temp_separator, thousands_separator)
            )
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df
