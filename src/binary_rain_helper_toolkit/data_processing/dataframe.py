import io
import logging
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
        if file_format == FileFormat.CSV:
            if file_format_options is None:
                dataframe = pd.read_csv(io.BytesIO(file_contents))
            else:
                dataframe = pd.read_csv(
                    io.BytesIO(file_contents), **file_format_options
                )
            dataframe = pd.read_csv(
                io.BytesIO(file_contents),
            )
        elif file_format == FileFormat.DICT:
            if file_format_options is None:
                dataframe = pd.DataFrame.from_dict(file_contents)
            else:
                dataframe = pd.DataFrame.from_dict(file_contents, **file_format_options)
        elif file_format == FileFormat.PARQUET:
            if file_format_options is None:
                dataframe = pd.read_parquet(io.BytesIO(file_contents), engine="pyarrow")
            else:
                dataframe = pd.read_parquet(
                    io.BytesIO(file_contents), engine="pyarrow", **file_format_options
                )
        elif file_format == FileFormat.JSON:
            if file_format_options is None:
                dataframe = pd.read_json(io.BytesIO(file_contents))
            else:
                dataframe = pd.read_json(
                    io.BytesIO(file_contents), **file_format_options
                )
        else:
            # other laod formats will be added later/when needed, e.g. load from json
            raise TypeError(
                f"Error creating dataframe. Unknown file format: {file_format}"
            )
    except Exception as exc:
        error_msg = f"Error creating dataframe. Exception: {exc}"
        logging.exception(error_msg)
        raise ValueError(error_msg) from exc
    return dataframe


def from_dataframe_to_type(
    dataframe: pd.DataFrame,
    file_format: FileFormat,
    file_format_options: dict | None = None,
) -> bytes:
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
        if file_format == FileFormat.CSV:
            if file_format_options is None:
                content = dataframe.to_csv(index=False)
            else:
                content = dataframe.to_csv(index=False, **file_format_options)
        elif file_format == FileFormat.DICT:
            if file_format_options is None:
                content = dataframe.to_dict(orient="records")
            else:
                content = dataframe.to_dict(**file_format_options)
        elif file_format == FileFormat.PARQUET:
            if file_format_options is None:
                content = dataframe.to_parquet(engine="pyarrow")
            else:
                content = dataframe.to_parquet(engine="pyarrow", **file_format_options)
        elif file_format == FileFormat.JSON:
            if file_format_options is None:
                content = dataframe.to_json()
            else:
                content = dataframe.to_json(**file_format_options)
        else:
            raise TypeError(
                f"Error converting dataframe. Unknown file format: {file_format}"
            )
    except Exception as exc:
        error_msg = f"Error converting dataframe. Exception: {exc}"
        logging.exception(error_msg)
        raise ValueError(error_msg) from exc
    return content


def merge_dataframes(
    df_history: pd.DataFrame | None,
    df_new: pd.DataFrame | None,
    sort: bool = False,
) -> pd.DataFrame:
    """
    Merge the new dataframe with the history dataframe.

    :param pd.DataFrame df_history:
        The history dataframe.
    :param pd.DataFrame df_new:
        The new dataframe to be merged.
    :param bool, optional sort:
        Sort the resulting dataframe. Default is False.

    :returns pandas.DataFrame df_full:
        The merged dataframe.
    exception : ValueError
        If an error occurs during dataframe merging
    """
    if isinstance(df_history, pd.DataFrame) and isinstance(df_new, pd.DataFrame):
        try:
            df_full = pd.concat([df_history, df_new], sort=sort)
        except Exception as exc:
            error_msg = f"Error merging dataframes. Exception: {exc}"
            logging.exception(error_msg)
            raise ValueError(error_msg) from exc
    else:
        error_msg = f"No dataframe provided for df_hsitory - got {type(df_history)} and/or df_new - got {type(df_new)}."  # pylint: disable=line-too-long
        logging.error(error_msg)
        raise ValueError(error_msg)
    return df_full
