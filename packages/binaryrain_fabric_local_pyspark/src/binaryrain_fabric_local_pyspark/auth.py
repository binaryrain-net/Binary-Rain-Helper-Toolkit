"""Create Azure credentials and access tokens for OneLake and SQL endpoint requests."""

from functools import cache

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import CredentialUnavailableError, DefaultAzureCredential

from binaryrain_fabric_local_pyspark.config import STORAGE_SCOPE
from binaryrain_fabric_local_pyspark.exceptions import FabricAuthenticationError


@cache
def _credential() -> DefaultAzureCredential:
    """Return the process-wide Azure credential used for Fabric access."""
    return DefaultAzureCredential()


def get_token(scope: str) -> str:
    """
    Return an Azure access token for ``scope``.

    :param str scope:
        OAuth scope the token is requested for, such as ``https://storage.azure.com/.default``.

    :returns str:
        Bearer token string.
    exception : FabricAuthenticationError
        If no credential source can provide a token, for example when the user is not
        signed in with ``az login``.
    """
    try:
        return _credential().get_token(scope).token
    except (ClientAuthenticationError, CredentialUnavailableError) as exc:
        raise FabricAuthenticationError(
            f"Could not get an Azure token for {scope}. Sign in with 'az login' or "
            f"configure the environment credentials. Details: {exc}"
        ) from exc


def _storage_token() -> str:
    """
    Return an Azure Storage bearer token for OneLake requests.

    :returns str:
        Bearer token string.
    exception : FabricAuthenticationError
        If no token can be obtained.
    """
    return get_token(STORAGE_SCOPE)
