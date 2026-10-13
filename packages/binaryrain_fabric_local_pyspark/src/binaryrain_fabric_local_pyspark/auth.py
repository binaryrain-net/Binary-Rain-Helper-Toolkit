"""Create Azure credentials and access tokens for OneLake requests."""

from functools import cache

from azure.identity import DefaultAzureCredential

from binaryrain_helper_local_pyspark.config import STORAGE_SCOPE


@cache
def _credential() -> DefaultAzureCredential:
    """Return the process-wide Azure credential used for OneLake access."""
    return DefaultAzureCredential()


def _storage_token() -> str:
    """Return an Azure Storage bearer token for OneLake requests."""
    return _credential().get_token(STORAGE_SCOPE).token
