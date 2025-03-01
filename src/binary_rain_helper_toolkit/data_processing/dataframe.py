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
):
    """
    Create a dataframe from the file contents.

    :param pandas.DataFrame dataframe:
        The dataframe to be converted.
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
                    dataframe = pd.DataFrame.from_dict(file_contents, **file_format_options)

            case FileFormat.PARQUET:
                if file_format_options is None:
                    dataframe = pd.read_parquet(io.BytesIO(file_contents), engine="pyarrow")
                else:
                    dataframe = pd.read_parquet(
                        io.BytesIO(file_contents), engine="pyarrow", **file_format_options
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
                    content = dataframe.to_parquet(engine="pyarrow", **file_format_options)

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
        raise ValueError(f"Error converting dataframe. See logs for more details. Exception: {exc}") from exc

    return content


def merge_dataframes(
    df_one: pd.DataFrame | None,
    df_two: pd.DataFrame | None,
    sort: bool = False,
) -> pd.DataFrame:
    """
    Merge two dataframes.

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
        raise ValueError(f"No dataframe provided for df_one - got {type(df_one)} and/or df_two - got {type(df_two)}.")

    return df_merged
