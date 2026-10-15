"""Tests for Azure credential and token handling."""

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import CredentialUnavailableError

from binaryrain_fabric_local_pyspark import auth as auth_module
from binaryrain_fabric_local_pyspark.config import STORAGE_SCOPE
from binaryrain_fabric_local_pyspark.exceptions import FabricAuthenticationError


@pytest.fixture
def credential_mock(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """Replace DefaultAzureCredential with a mock and reset the cached credential."""
    credential = MagicMock()
    credential.get_token.return_value.token = "secret-token"
    monkeypatch.setattr(auth_module, "DefaultAzureCredential", MagicMock(return_value=credential))
    auth_module._credential.cache_clear()
    yield credential
    auth_module._credential.cache_clear()


def test_credential_is_created_once(credential_mock: MagicMock) -> None:
    """The credential is cached for the whole process."""
    assert auth_module._credential() is auth_module._credential()
    auth_module.DefaultAzureCredential.assert_called_once_with()


def test_get_token_returns_token_for_scope(credential_mock: MagicMock) -> None:
    """get_token returns the token string for the requested scope."""
    assert auth_module.get_token("scope/.default") == "secret-token"
    credential_mock.get_token.assert_called_once_with("scope/.default")


@pytest.mark.parametrize(
    "error",
    [ClientAuthenticationError("rejected"), CredentialUnavailableError("not signed in")],
)
def test_get_token_wraps_credential_errors(credential_mock: MagicMock, error: Exception) -> None:
    """Credential errors are raised as FabricAuthenticationError."""
    credential_mock.get_token.side_effect = error

    with pytest.raises(FabricAuthenticationError, match="az login") as exc_info:
        auth_module.get_token("scope/.default")

    assert exc_info.value.__cause__ is error


def test_storage_token_uses_storage_scope(credential_mock: MagicMock) -> None:
    """_storage_token requests a token for the Azure Storage scope."""
    assert auth_module._storage_token() == "secret-token"
    credential_mock.get_token.assert_called_once_with(STORAGE_SCOPE)
