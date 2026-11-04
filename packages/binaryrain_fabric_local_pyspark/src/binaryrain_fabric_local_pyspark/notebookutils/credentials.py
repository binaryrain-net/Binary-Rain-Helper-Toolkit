"""Read Azure Key Vault secrets using the local Azure identity."""

import re
from urllib.parse import urlsplit

from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.identity import CredentialUnavailableError
from azure.keyvault.secrets import SecretClient

from binaryrain_fabric_local_pyspark.auth import _credential
from binaryrain_fabric_local_pyspark.exceptions import (
    FabricAuthenticationError,
    FabricConfigurationError,
    FabricSecretError,
)


def getSecret(akvName: str, secret: str) -> str:
    """
    Return the latest secret value from an Azure Key Vault HTTPS endpoint.

    Uses the package's DefaultAzureCredential, including Azure CLI credentials.
    Secret values are not cached, logged or automatically redacted from notebook output.

    :param str akvName:
        Key Vault endpoint, such as ``https://myvault.vault.azure.net/``.
    :param str secret:
        Secret name, containing 1 to 127 letters, digits or hyphens.

    :returns str:
        The plaintext secret value, including an empty string when stored that way.
    exception : FabricConfigurationError
        If the endpoint or secret name is invalid.
    exception : FabricAuthenticationError
        If Azure credentials are unavailable or authentication fails.
    exception : FileNotFoundError
        If the vault or secret does not exist.
    exception : FabricSecretError
        If access is denied, a request fails or the response has no secret value.
    """
    if not isinstance(akvName, str) or not akvName.strip():
        raise FabricConfigurationError("A Key Vault HTTPS endpoint is required.")
    try:
        endpoint = urlsplit(akvName)
    except ValueError as exc:
        raise FabricConfigurationError("Invalid Key Vault HTTPS endpoint.") from exc
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.path not in ("", "/")
        or endpoint.query
        or endpoint.fragment
    ):
        raise FabricConfigurationError("Use the Key Vault HTTPS endpoint without a secret path.")
    if not isinstance(secret, str) or re.fullmatch(r"[A-Za-z0-9-]{1,127}", secret) is None:
        raise FabricConfigurationError(
            "A secret name must contain 1 to 127 letters, digits or hyphens."
        )

    try:
        with SecretClient(vault_url=akvName, credential=_credential()) as client:
            value = client.get_secret(secret).value
    except (ClientAuthenticationError, CredentialUnavailableError) as exc:
        raise FabricAuthenticationError(
            "Could not authenticate to Key Vault. Sign in with 'az login' or configure "
            "environment credentials."
        ) from exc
    except ResourceNotFoundError as exc:
        raise FileNotFoundError("The Key Vault or requested secret does not exist.") from exc
    except (HttpResponseError, ServiceRequestError, ServiceResponseError) as exc:
        raise FabricSecretError(
            "Could not read the Key Vault secret. Check secret Get permissions and "
            "network access to the vault."
        ) from exc

    if value is None:
        raise FabricSecretError("Key Vault returned no secret value.")
    return value
