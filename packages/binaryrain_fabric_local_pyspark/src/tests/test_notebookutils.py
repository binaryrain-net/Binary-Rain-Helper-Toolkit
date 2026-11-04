"""Tests for the local notebookutils credentials replacement."""

import json
import subprocess
import sys
from functools import partial
from unittest.mock import MagicMock

import pytest
from azure.core.credentials import AccessToken, TokenCredential
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.core.pipeline.transport import HttpRequest, HttpResponse, HttpTransport
from azure.identity import CredentialUnavailableError
from azure.keyvault.secrets import SecretClient

from binaryrain_fabric_local_pyspark import notebookutils
from binaryrain_fabric_local_pyspark.exceptions import (
    FabricAuthenticationError,
    FabricConfigurationError,
    FabricSecretError,
)
from binaryrain_fabric_local_pyspark.notebookutils import credentials

VAULT_URL = "https://example.vault.azure.net/"


@pytest.fixture
def client_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the SDK client and credential without contacting Azure."""
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_secret.return_value.value = "synthetic-value"
    monkeypatch.setattr(credentials, "SecretClient", MagicMock(return_value=client))
    monkeypatch.setattr(credentials, "_credential", MagicMock())
    return client


def test_notebookutils_exposes_credentials() -> None:
    """The package import exposes the credentials module."""
    assert notebookutils.credentials is credentials


@pytest.mark.parametrize("value", ["synthetic-value", "", '{"setting": 1}'])
def test_get_secret_returns_plaintext(client_mock: MagicMock, value: str) -> None:
    """Secret values remain strings and the SDK client is closed after each call."""
    client_mock.get_secret.return_value.value = value

    assert notebookutils.credentials.getSecret(akvName=VAULT_URL, secret="db-password") == value

    credentials.SecretClient.assert_called_once_with(
        vault_url=VAULT_URL, credential=credentials._credential.return_value
    )
    client_mock.get_secret.assert_called_once_with("db-password")
    client_mock.__exit__.assert_called_once()


def test_secret_values_are_not_cached(client_mock: MagicMock) -> None:
    """Repeated reads observe changed values instead of using the data cache."""
    client_mock.get_secret.side_effect = [
        MagicMock(value="first-synthetic-value"),
        MagicMock(value="rotated-synthetic-value"),
    ]

    assert credentials.getSecret(VAULT_URL, "db-password") == "first-synthetic-value"
    assert credentials.getSecret(VAULT_URL, "db-password") == "rotated-synthetic-value"
    assert client_mock.get_secret.call_count == 2
    assert credentials.SecretClient.call_count == 2


@pytest.mark.parametrize(
    "vault_url",
    [
        None,
        1,
        "",
        " ",
        "myvault",
        "http://example.vault.azure.net/",
        "https://",
        "https://user:password@example.vault.azure.net/",
        "https://example.vault.azure.net/secrets/name",
        "https://example.vault.azure.net/?query=value",
        "https://example.vault.azure.net/#fragment",
        "https://[invalid",
    ],
)
def test_invalid_endpoint_never_requests_credentials(
    client_mock: MagicMock, vault_url: object
) -> None:
    """Invalid endpoints fail before a client or credential is created."""
    with pytest.raises(FabricConfigurationError, match="Key Vault"):
        credentials.getSecret(vault_url, "db-password")

    credentials._credential.assert_not_called()
    credentials.SecretClient.assert_not_called()


@pytest.mark.parametrize("name", [None, 1, "", " ", "bad/name", "bad_name", "a" * 128])
def test_invalid_secret_name_never_contacts_azure(client_mock: MagicMock, name: object) -> None:
    """Invalid secret names fail before requesting credentials."""
    with pytest.raises(FabricConfigurationError, match="secret name"):
        credentials.getSecret(VAULT_URL, name)

    credentials._credential.assert_not_called()
    credentials.SecretClient.assert_not_called()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ClientAuthenticationError("synthetic authentication failure"), FabricAuthenticationError),
        (CredentialUnavailableError("synthetic unavailable credential"), FabricAuthenticationError),
        (ResourceNotFoundError("synthetic missing secret"), FileNotFoundError),
        (HttpResponseError("synthetic forbidden response"), FabricSecretError),
        (ServiceRequestError("synthetic network failure"), FabricSecretError),
        (ServiceResponseError("synthetic response failure"), FabricSecretError),
    ],
)
def test_azure_errors_propagate_with_their_cause(
    client_mock: MagicMock, error: Exception, expected: type[Exception]
) -> None:
    """Failures produce explicit exceptions and still close the client."""
    client_mock.get_secret.side_effect = error

    with pytest.raises(expected) as exc_info:
        credentials.getSecret(VAULT_URL, "db-password")

    assert exc_info.value.__cause__ is error
    client_mock.__exit__.assert_called_once()


def test_missing_value_is_not_a_success(client_mock: MagicMock) -> None:
    """A response with no value raises instead of returning None or an empty fallback."""
    client_mock.get_secret.return_value.value = None

    with pytest.raises(FabricSecretError, match="no secret value"):
        credentials.getSecret(VAULT_URL, "db-password")


class _JsonResponse(HttpResponse):
    """An in-memory HTTP response for the real Azure SDK pipeline."""

    def __init__(self, request: HttpRequest, status: int, payload: dict[str, object]) -> None:
        """Build a synthetic JSON response without a network connection."""
        super().__init__(request, None)
        self.status_code = status
        self.headers = {"content-type": "application/json"}
        self.content_type = "application/json"
        self.reason = "Synthetic response"
        self._payload = payload
        self._body = json.dumps(payload).encode("utf-8")

    def body(self) -> bytes:
        """Return the serialized response body."""
        return self._body

    def json(self) -> dict[str, object]:
        """Return the decoded JSON body for current Azure SDK clients."""
        return self._payload


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, None),
        (401, FabricAuthenticationError),
        (403, FabricSecretError),
        (404, FileNotFoundError),
        (500, FabricSecretError),
    ],
)
def test_real_sdk_serialization_and_http_errors(
    monkeypatch: pytest.MonkeyPatch, status: int, expected: type[Exception] | None
) -> None:
    """Exercise the actual SDK client against synthetic HTTP responses."""
    transport = MagicMock(spec=HttpTransport)
    credential = MagicMock(spec=TokenCredential)
    credential.get_token.return_value = AccessToken("synthetic-token", 2147483647)
    payload = (
        {"value": "synthetic-value", "id": f"{VAULT_URL}secrets/db-password/version"}
        if status == 200
        else {"error": {"code": "SyntheticFailure", "message": "Synthetic service failure"}}
    )

    def send(request: HttpRequest, **kwargs: object) -> HttpResponse:
        """Supply JSON responses at the SDK HTTP transport boundary."""
        if transport.send.call_count == 1:
            response = _JsonResponse(request, 401, {})
            response.headers["WWW-Authenticate"] = (
                'Bearer authorization="https://login.microsoftonline.com/'
                '00000000-0000-0000-0000-000000000000", resource="https://vault.azure.net"'
            )
            return response
        return _JsonResponse(request, status, payload)

    transport.send.side_effect = send
    monkeypatch.setattr(credentials, "_credential", lambda: credential)
    monkeypatch.setattr(
        credentials, "SecretClient", partial(SecretClient, transport=transport, retry_total=0)
    )

    if expected is None:
        assert credentials.getSecret(VAULT_URL, "db-password") == "synthetic-value"
    else:
        with pytest.raises(expected):
            credentials.getSecret(VAULT_URL, "db-password")

    request = transport.send.call_args.args[0]
    assert request.method == "GET"
    assert request.url.startswith(f"{VAULT_URL}secrets/db-password/?api-version=")
    assert transport.send.call_count == 2
    assert credential.get_token.call_args.args == ("https://vault.azure.net/.default",)
    transport.__exit__.assert_called_once()


def test_import_does_not_require_spark_or_fabric_configuration() -> None:
    """The replacement imports without loading PySpark or starting an Azure credential."""
    script = """
import os
import sys
for name in list(os.environ):
    if name.startswith("FABRIC_"):
        del os.environ[name]
sys.modules["pyspark"] = None
from binaryrain_fabric_local_pyspark import notebookutils
from binaryrain_fabric_local_pyspark.auth import _credential
assert callable(notebookutils.credentials.getSecret)
assert _credential.cache_info().currsize == 0
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
