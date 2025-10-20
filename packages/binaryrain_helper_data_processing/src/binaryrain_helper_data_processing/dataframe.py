import io
from enum import Enum
import math
import re
import pandas as pd
import warnings
import functools


def deprecated(reason: str = None):
    """Decorator to mark functions as deprecated."""

    def decorator(func):
        message = f"Function {func.__name__} is deprecated."
        if reason:
            message += f" {reason}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


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
        The format of the file to be loaded.
        Currently supported: `csv` and `dict`, `parquet`, `json`.
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
                    dataframe = pd.read_csv(io.BytesIO(file_contents), **file_format_options)

            case FileFormat.DICT:
                if file_format_options is None:
                    dataframe = pd.DataFrame.from_dict(file_contents)
                else:
                    dataframe = pd.DataFrame.from_dict(file_contents, **file_format_options)

            case FileFormat.PARQUET:
                if file_format_options is None:
                    dataframe = pd.read_parquet(io.BytesIO(file_contents), engine="pyarrow")
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
                    dataframe = pd.read_json(io.BytesIO(file_contents), **file_format_options)

            case _:
                raise TypeError(f"Error creating dataframe. Unknown file format: {file_format}")
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
                    content = dataframe.to_parquet(engine="pyarrow", **file_format_options)

            case FileFormat.JSON:
                if file_format_options is None:
                    content = dataframe.to_json()
                else:
                    content = dataframe.to_json(**file_format_options)

            case _:
                raise TypeError(f"Error converting dataframe. Unknown file format: {file_format}")
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
            f"No dataframe provided for df_one - got {type(df_one)} "
            f"and/or df_two - got {type(df_two)}."
        )

    return df_merged


def convert_to_datetime(
    df: pd.DataFrame,
    date_formats: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert object columns in a dataframe to datetime dtype where possible.
    If no date_formats are provided, a set of common formats are being used
    and inference is enabled as a first step.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame whose object columns may be converted in place.
    date_formats : list[str] | None, default=None
        Ordered list of strftime-compatible date format strings to attempt.
        If None, a built-in set of common formats is used and inference is
        enabled as a first step.
    Returns
    -------
    pd.DataFrame
        The same DataFrame instance with successfully parsed columns converted
        to datetime dtype.
    """
    infer = False
    if date_formats is None:
        date_formats = [
            "%d.%m.%Y",
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ]
        infer = True

    for column in df.columns:
        if df[column].dtype == "object":
            if df[column].isna().all():
                continue

            # Try inference first (fastest for common formats)
            if infer:
                try:
                    converted = pd.to_datetime(df[column], errors="coerce")
                    if converted.notna().sum() > len(df[column]) * 0.5:  # >50% success
                        df[column] = converted
                        continue
                except (ValueError, TypeError):
                    pass

            # Fall back to explicit formats
            for date_format in date_formats:
                try:
                    converted = pd.to_datetime(df[column], format=date_format, errors="coerce")
                    if converted.notna().any():
                        df[column] = converted
                        break
                except (ValueError, TypeError):
                    continue

    return df


def format_datetime_columns(
    df: pd.DataFrame,
    datetime_columns: list[str],
    datetime_format: str,
    datetime_string_columns: list[str] | None = None,
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
        The columns to format as datetime strings. Optional.
        If not provided, the same columns as datetime_columns will be used.

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
        try:
            df[datetime_string_columns[i]] = pd.to_datetime(df[datetime_columns[i]]).dt.strftime(
                datetime_format
            )
        except Exception as e:
            raise ValueError(f"Error formatting column {datetime_columns[i]}: {e}") from e
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a dataframe by removing rows with missing values.

    :param pd.DataFrame df:
        The dataframe to clean.

    :returns pd.DataFrame df:
        The cleaned dataframe
    """
    prepared_df = df.replace("nan", pd.NA)
    prepared_df = prepared_df.replace("", pd.NA)
    prepared_df = prepared_df.dropna()
    prepared_df = prepared_df.reset_index(drop=True)
    return prepared_df


def remove_empty_values(
    df: pd.DataFrame,
    filter_column: str,
    dropna: bool = True,
    reset_index: bool = True,
) -> pd.DataFrame:
    """
    Return rows where filter_column is (optionally) non-null and not an empty/blank string.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    filter_column : str
        Column on which empties are filtered.
    dropna : bool, default True
        If True, drop rows where filter_column is NaN.
    reset_index : bool, default True
        If True, reset the index in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if filter_column not in df.columns:
        raise KeyError(f"Column '{filter_column}' not found in DataFrame.")

    col = df[filter_column]

    # Start with all True
    mask = pd.Series(True, index=df.index)

    if dropna:
        mask &= col.notna()

    # Only apply string trimming if dtype is string-like
    if col.dtype == "object":
        # Strip and keep rows where stripped value != ""
        stripped = col.astype(str).str.strip()
        mask &= stripped.ne("")
    else:
        # For non-string dtypes only NaN check is relevant
        pass

    result = df.loc[mask]

    if reset_index:
        result = result.reset_index(drop=True)

    return result


@deprecated("This function has been deprecated. Please use `format_numeric_to_string` instead.")
def format_numeric_values(
    df: pd.DataFrame,
    columns: list[str],
    swap_separators: bool = False,
    decimal_separator: str = ",",
    thousands_separator: str = ".",
    old_decimal_separator: str = ".",
    old_thousands_separator: str = ",",
    temp_separator: str = "|",
) -> pd.DataFrame:
    """
    Format numeric values in a dataframe.
    Additionally it swaps the decimal and thousands separators.
    This is useful when the data is read from a file with a different locale.

    :param pd.DataFrame df:
        The dataframe to format numeric values in.
    :param list[str] columns:
        The columns to format as numeric values.
    :param bool swap_separators:
        Swap the decimal and thousands separators. Default is False.
    :param str decimal_separator:
            The decimal separator to use. Default is `,`.
    :param str thousands_separator:
                The thousands separator to use. Default is `.`.
    :param str old_decimal_separator:
        The old decimal separator to replace. Default is `.`.
    :param str old_thousands_separator:
        The old thousands separator to replace. Default is `,`.
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


def format_numeric_to_string(
    df: pd.DataFrame,
    columns: list[str],
    decimal_separator: str = ",",
    thousands_separator: str = ".",
    old_decimal_separator: str = ".",
    old_thousands_separator: str = ",",
    temp_separator: str = "|",
    decimal_places: int = 2,
) -> pd.DataFrame:
    """
    Format specified columns as locale-style numeric strings.

    :param pd.DataFrame df:
        The dataframe to format numeric values in.
    :param list[str] columns:
        The columns to format as numeric values.
    :param str decimal_separator:
            The decimal separator to use. Default is `,`.
    :param str thousands_separator:
                The thousands separator to use. Default is `.`.
    :param str old_decimal_separator:
        The old decimal separator to replace. Default is `,`.
    :param str old_thousands_separator:
        The old thousands separator to replace. Default is `.`.
    :param str temp_separator:
        The temporary separator to use. Default is `|`.
    :param int decimal_places:
        Number of decimal places to format to. Default is 2.

    Mutates and returns the same DataFrame.
    """
    if not columns:
        return df

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataframe: {missing}")

    if decimal_places < 0:
        raise ValueError("decimal_places must be >= 0")

    if decimal_separator == thousands_separator:
        raise ValueError("decimal_separator and thousands_separator must differ")

    if temp_separator in (decimal_separator, thousands_separator):
        raise ValueError("temp_separator must differ from decimal and thousands separators")

    # Helper function for manual rounding to avoid banker's rounding
    def _round_manual(x, decimals=0):
        factor = 10**decimals
        # Add 0.5 for positive numbers or -0.5 for negatives to emulate rounding
        if x >= 0:
            return math.floor(x * factor + 0.5) / factor
        else:
            return math.ceil(x * factor - 0.5) / factor

    # Precompile regex-safe replacements
    if old_thousands_separator:
        old_thousands_pattern = re.escape(old_thousands_separator)
    else:
        old_thousands_pattern = None
    if old_decimal_separator:
        old_decimal_pattern = re.escape(old_decimal_separator)
    else:
        old_decimal_pattern = None

    fmt = f"{{:,.{decimal_places}f}}"

    for column in columns:
        series = df[column]

        # Parse to numeric
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
        else:
            prepared_series = series.astype(str).str.strip()
            if old_thousands_pattern:
                prepared_series = prepared_series.str.replace(old_thousands_pattern, "", regex=True)
            if old_decimal_pattern and old_decimal_separator != ".":
                prepared_series = prepared_series.str.replace(old_decimal_pattern, ".", regex=True)
            # Empty strings -> NaN
            prepared_series = prepared_series.replace({"": None})
            numeric = pd.to_numeric(prepared_series, errors="coerce")

        # Format numbers; NaN -> empty string
        formatted = numeric.map(
            lambda v: fmt.format(_round_manual(v, decimal_places)) if pd.notna(v) else ""
        )

        # check if any formatting resulted in non-empty strings
        if formatted.eq("").all():
            df[column] = formatted
            continue

        # Adjust separators according to defined separators
        formatted = (
            formatted.str.replace(",", temp_separator, regex=False)
            .str.replace(".", decimal_separator, regex=False)
            .str.replace(temp_separator, thousands_separator, regex=False)
        )

        df[column] = formatted

    return df
