import io
import logging
import pandas as pd

def create_dataframe(file_contents: bytes, file_format: str):
    """Create a dataframe from the file contents."""
    try:
        if file_format == "csv":
            dataframe = pd.read_csv(io.BytesIO(file_contents))
        elif file_format == "dict":
            dataframe = pd.DataFrame.from_dict(file_contents)
        else:
            # other laod formats will be added later/when needed, e.g. load from json
            raise TypeError(f"Error creating dataframe. Unknown file format: {file_format}")
    except Exception as exc:
        error_msg = f"Error creating dataframe. Exception: {exc}"
        logging.exception(error_msg)
        raise ValueError(error_msg) from exc
    return dataframe

def merge_dataframes(df_history, df_new):
    """Merge the new dataframe with the history dataframe."""
    if isinstance(df_history, pd.DataFrame) and isinstance(df_new, pd.DataFrame):
        try:
            df_full = pd.concat([df_history, df_new], sort=False)
        except Exception as exc:
            error_msg = f"Error merging dataframes. Exception: {exc}"
            logging.exception(error_msg)
            raise ValueError(error_msg) from exc
    else:
        error_msg = f"No dataframe provided for df_hsitory - got {type(df_history)} and/or df_new - got {type(df_new)}." #pylint: disable=line-too-long
        logging.error(error_msg)
        raise ValueError(error_msg)

    return df_full