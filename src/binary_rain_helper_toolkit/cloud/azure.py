import json
import pandas as pd
import logging

import azure.functions as func
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from data_processing.dataframe import (
    create_dataframe,
    from_dataframe_to_type,
    FileFormat,
)


def return_http_response(message: str, status_code: int) -> func.HttpResponse:
    """
    Format an HTTP response.

    ### Parameters:
    ----------
    message : str
        The message to be returned in the response.
    status_code : int
        The status code of the response.

    ### Returns:
    ----------
    response : func.HttpResponse
        The formatted HTTP response
    """
    if str(status_code).startswith("2"):
        status = "OK"
    else:
        status = "NOK"

    return func.HttpResponse(
        json.dumps({"response": message, "status": status}),
        status_code=status_code,
        mimetype="application/json",
    )


def read_blob_data(
    blob_account: str,
    container_name: str,
    blob_name: str,
    file_format: FileFormat = FileFormat.PARQUET,
) -> pd.DataFrame:
    """
    Read data from a blob storage account.

    ### Parameters:
    ----------
    blob_account : str
        The name of the blob account.
    container_name : str
        The name of the container.
    blob_name : str
        The name of the blob.
    file_format : FileFormat, optional
        The format of the file to be loaded. Default is FileFormat.PARQUET.

    ### Returns:
    ----------
    df : pd.DataFrame
        The data read from the blob.
    exception : ValueError
        If an error occurs while trying to download the blob data.
    """
    # Initialize the dataframe
    df = pd.DataFrame()

    # validate the input parameters
    if not blob_account or not container_name or not blob_name:
        error_msg = "Error: Missing required input parameters."
        logging.error(error_msg)
        return ValueError(error_msg)

    try:
        # Create blob service client
        blob_service_client = BlobServiceClient(
            blob_account,
            credential=DefaultAzureCredential(),
        )

        # Create container service client
        blob_client = blob_service_client.get_blob_client(
            container_name,
            blob_name,
        )

        chunks_data = blob_client.download_blob()
        chunk_list = []

        # Download the data in chunks.
        # This is useful for large files, since Files over 35MB can cause issues.
        for chunk in chunks_data.chunks():
            chunk_list.append(chunk)

        # Combine the chunks into a single byte array
        byte_array = b"".join(chunk_list)

        # Load data into a pandas dataframe
        df = create_dataframe(byte_array, file_format)
    except Exception as e:  # pylint: disable=broad-except
        error_msg = "Error while trying to download the blob data. Exception: %s", e
        logging.exception(error_msg)
        return return_http_response(error_msg, 500)
    return df


def upload_blob_data(
    blob_account: str,
    container_name: str,
    blob_name: str,
    df: pd.DataFrame,
    file_format: FileFormat = FileFormat.PARQUET,
    file_format_options: dict = None,
) -> pd.DataFrame:
    """
    Upload data from a blob storage account.

    ### Parameters:
    ----------
    blob_account : str
        The name of the blob account.
    container_name : str
        The name of the container.
    blob_name : str
        The name of the blob.
    file_format : FileFormat, optional
        The format of the file to be loaded. Default is FileFormat.PARQUET.
    file_format_options : dict, optional
        The format options for the file. Default is None.

    ### Returns:
    ----------
    df : pd.DataFrame
        The data read from the blob.
    exception : ValueError
        If an error occurs while trying to download the blob data.

    ### Example:
    ----------
    ```python
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    file_format_options = {
        'compression': 'snappy',
        'use_deprecated_int96_timestamps': True
    }
    upload_blob_data(
        blob_account='blob_account',
        container_name='container_name',
        blob_name='blob_name',
        df=df,
        file_format=FileFormat.PARQUET,
        file_format_options=file_format_options
    )
    ```
    """
    # Initialize the dataframe
    df = pd.DataFrame()

    # validate the input parameters
    if not blob_account or not container_name or not blob_name:
        error_msg = "Error: Missing required input parameters."
        logging.error(error_msg)
        return ValueError(error_msg)

    try:
        # Create blob service client
        blob_service_client = BlobServiceClient(
            blob_account,
            credential=DefaultAzureCredential(),
        )

        # Create container service client
        blob_client = blob_service_client.get_blob_client(
            container_name,
            blob_name,
        )

        # Upload the fixed dataframe to the blob
        blob = from_dataframe_to_type(
            df, file_format, file_format_options=file_format_options
        )
        blob_client.upload_blob(
            blob,
            overwrite=True,
            blob_type="BlockBlob",
            length=len(blob),
            connection_timeout=800,
        )
    except Exception as e:  # pylint: disable=broad-except
        error_msg = "Error while trying to download the blob data. Exception: %s", e
        logging.exception(error_msg)
        return return_http_response(error_msg, 500)
    return df
