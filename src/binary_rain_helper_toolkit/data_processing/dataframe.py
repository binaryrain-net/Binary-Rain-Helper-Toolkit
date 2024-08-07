import io
import logging
import pandas as pd


def create_dataframe(
    file_contents: bytes | dict,
    file_format: str,
    seperator: str = None,
):
    """
    Create a dataframe from the file contents.

    ### Parameters:
    ----------
    file_contents : bytes | dict
        The contents of the file to be loaded.
    file_format : str
        The format of the file to be loaded. Currently supported: "csv" and "dict".
    seperator : str, optional
        The seperator used in the csv file. Default is None.

    ### Returns:
    ----------
    dataframe : pd.DataFrame
        The dataframe created from the file contents
    exception : ValueError
        If an error occurs during dataframe creation
    """
    try:
        if file_format == "csv":
            dataframe = pd.read_csv(io.BytesIO(file_contents), sep=seperator)
        elif file_format == "dict":
            dataframe = pd.DataFrame.from_dict(file_contents)
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


def merge_dataframes(
    df_history: pd.DataFrame | None,
    df_new: pd.DataFrame | None,
    sort: bool = False,
) -> pd.DataFrame:
    """
    Merge the new dataframe with the history dataframe.

    ### Parameters:
    ----------
    df_history : pd.DataFrame
        The history dataframe.
    df_new : pd.DataFrame
        The new dataframe to be merged.
    sort : bool, optional
        Sort the resulting dataframe. Default is False.

    ### Returns:
    ----------
    df_full : pd.DataFrame
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
