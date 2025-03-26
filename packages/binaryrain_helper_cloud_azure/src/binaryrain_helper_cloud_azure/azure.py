import json

import azure.functions as func
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.keyvault.secrets import SecretClient


def return_http_response(message: str, status_code: int) -> func.HttpResponse:
    """
    Format an HTTP response.

    :param str message:
        The message to be returned in the response.
    :param int status_code:
        The status code of the response.

    :returns azure.functions.HttpResponse:
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


def read_blob_data(blob_account: str, container_name: str, blob_name: str) -> bytes:
    """
    Read data from a blob storage account.

    :param str blob_account:
        The name of the blob account.
    :param str container_name:
        The name of the container.
    :param str blob_name:
        The name of the blob.

    :returns file_obj : bytes
        content of the object as bytes.
    exception : ValueError
        If an error occurs while trying to download the blob data.
    """

    # validate the input parameters
    if not blob_account or not container_name or not blob_name:
        raise ValueError("Error: Missing required input parameters.")

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

    except Exception as e:  # pylint: disable=broad-except
        raise ValueError(
            f"Error while trying to download the blob data. Exception: {e}"
        )

    return byte_array


def upload_blob_data(
    blob_account: str, container_name: str, blob_name: str, file_contents: bytes
) -> bool:
    """
    Save file to a storage account / blob.

    :param str blob_account:
        The name of the blob account.
    :param str container_name:
        The name of the container.
    :param str blob_name:
        The name of the blob.
    :param bytes file_contents:
        The file contents to be saved.

    :returns bool
        True if the file was saved successfully.
    exception : Exception
        exception if the file cannot be saved.

    Example
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

    # validate the input parameters
    if not blob_account:
        raise ValueError("No blob account provided.")
    if not container_name:
        raise ValueError("No container name provided.")
    if not blob_name:
        raise ValueError("No blob name provided.")
    if (
        not file_contents
        or not isinstance(file_contents, bytes)
        or len(file_contents) == 0
    ):
        raise ValueError(
            "No file contents provided or file contents are empty or not of type bytes."
        )

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

        # Upload to the blob
        blob_client.upload_blob(
            file_contents,
            overwrite=True,
            blob_type="BlockBlob",
            length=len(file_contents),
        )
    except Exception as e:  # pylint: disable=broad-except
        raise ValueError(f"Error while trying to upload the blob data. Exception: {e}")

    return True


def get_secret_data(key_vault_url: str, secret_name: str) -> dict:
    """
    Get secret data from Azure Key Vault.

    :param str key_vault_url:
        The URL of the Azure Key Vault.
    :param str secret_name:
        The name of the secret.

    :returns dict:
        The secret data.
    exception: ValueError
        The exception raised if an error occurs while trying to access the secret data.
    """

    # validate the input parameters
    if not key_vault_url:
        raise ValueError("No Key Vault URL provided.")
    if not secret_name:
        raise ValueError("No secret name provided.")

    try:
        # Create a SecretClient
        secret_client = SecretClient(
            vault_url=key_vault_url,
            credential=DefaultAzureCredential(),
        )
    except Exception as e:  # pylint: disable=broad-except
        raise ValueError(f"Error creating SecretClient. Exception: {e}")

    try:
        secret_data = secret_client.get_secret(secret_name).value
    except Exception as e:  # pylint: disable=broad-except
        raise ValueError(f"Error while trying to get the secret data. Exception: {e}")

    return secret_data
