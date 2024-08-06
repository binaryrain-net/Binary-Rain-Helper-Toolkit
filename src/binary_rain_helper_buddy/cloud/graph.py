import logging
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import msal

def get_graph_api_access_token(
        key_vault_url: str,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        credential: DefaultAzureCredential = None,
        graph_scope: str = "https://graph.microsoft.com/.default"
) -> str :
    """retrieve access token for Microsoft Graph API from Azure Key Vault"""

    if credential is None:
        # default credential will be managed identity if deployed to Azure
        # to run locally first use az login to authenticate with Azure
        credential = DefaultAzureCredential()
    client = SecretClient(vault_url=key_vault_url, credential=credential)

    # Checks if client is set
    if client is None:
        logging.error("Error creating DefaultAzureCredential client.")
        raise ValueError("Error creating DefaultAzureCredential client.")

    # Setup MSAL configuration
    client_id = client.get_secret(client_id)
    client_secret = client.get_secret(client_secret)
    tenant_id = client.get_secret(tenant_id)

    if client_id is None:
        raise ValueError("client_id is not set.")

    if client_secret is None:
        raise ValueError("client_secret is not set.")

    if tenant_id is None:
        raise ValueError("tenant_id is not set.")

    # Create a preferably long-lived app instance which maintains a token cache.
    logging.debug(
        "Creating a preferably long-lived app instance which maintains a token cache....."
    )

    msal_app = msal.ConfidentialClientApplication(
        client_id.value,
        authority="https://login.microsoftonline.com/" + (tenant_id.value or ""),
        client_credential=client_secret.value,
    )

    # Checks if app is set
    if msal_app is None:
        logging.error("Error setting up MSAL client.")
        raise ValueError("Error setting up MSAL client.")

    # check if a token existis in cache and if not, get a new one from AAD
    result = msal_app.acquire_token_silent(graph_scope, account=None)

    if not result:
        logging.debug("No suitable token exists in cache. Let's get a new one from AAD.")
        try:
            result = msal_app.acquire_token_for_client(
                scopes=[graph_scope]
                )
        except ValueError as e:
            logging.error("Error auqiring a new token Exception: %s", e)
            raise ValueError("Error auqiring a new token.") from e

    if result is None or "access_token" not in result:
        logging.error("Error retrieving access token. Got response: %s", result)
        raise ValueError("Error retrieving access token.")

    logging.debug("Access token successfully retrieved.")
    return result["access_token"]