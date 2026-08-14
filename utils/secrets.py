"""
Secure secrets management.

Design goal (Security by Design): API keys are NEVER hardcoded and never
logged. This module provides one unified interface, `SecretsProvider`,
behind which any backend (plain environment variables, HashiCorp Vault,
AWS Secrets Manager) can be swapped without touching agent or LLM-client
code. The backend is chosen via the `IDAMP_SECRETS_BACKEND` env var.
"""

from __future__ import annotations

import os
import json
import functools
from abc import ABC, abstractmethod
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=False)


class SecretNotFoundError(RuntimeError):
    """Raised when a requested secret/key cannot be located in the backend."""


class SecretsProvider(ABC):
    """Unified interface every secrets backend must implement."""

    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        """Return the secret value for `key`, or None if not present."""
        raise NotImplementedError

    def require_secret(self, key: str) -> str:
        value = self.get_secret(key)
        if not value:
            raise SecretNotFoundError(
                f"Required secret '{key}' was not found. Set it via the "
                f"configured secrets backend (see .env.example)."
            )
        return value

    def __repr__(self) -> str:
        # Never expose values in repr/logging.
        return f"<{self.__class__.__name__}>"


class EnvSecretsProvider(SecretsProvider):
    """Default backend: plain environment variables (from process env or .env)."""

    def get_secret(self, key: str) -> Optional[str]:
        return os.environ.get(key)


class AWSSecretsManagerProvider(SecretsProvider):
    """Backend that reads a JSON blob of key/value secrets from AWS Secrets Manager."""

    def __init__(self, secret_id: Optional[str] = None, region: Optional[str] = None) -> None:
        self._secret_id = secret_id or os.environ.get("AWS_SECRETS_MANAGER_SECRET_ID")
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._cache: Optional[dict] = None

    def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        if not self._secret_id:
            raise SecretNotFoundError("AWS_SECRETS_MANAGER_SECRET_ID is not configured.")
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "boto3 is required for the aws_secrets_manager backend. "
                "Install it via `pip install boto3`."
            ) from exc

        client = boto3.client("secretsmanager", region_name=self._region)
        response = client.get_secret_value(SecretId=self._secret_id)
        raw = response.get("SecretString") or "{}"
        self._cache = json.loads(raw)
        return self._cache

    def get_secret(self, key: str) -> Optional[str]:
        try:
            return self._load().get(key)
        except SecretNotFoundError:
            return None


class VaultSecretsProvider(SecretsProvider):
    """Backend that reads secrets from HashiCorp Vault (KV v2)."""

    def __init__(self, addr: Optional[str] = None, token: Optional[str] = None,
                 secret_path: Optional[str] = None) -> None:
        self._addr = addr or os.environ.get("VAULT_ADDR")
        self._token = token or os.environ.get("VAULT_TOKEN")
        self._secret_path = secret_path or os.environ.get("VAULT_SECRET_PATH")
        self._cache: Optional[dict] = None

    def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        if not (self._addr and self._token and self._secret_path):
            raise SecretNotFoundError("VAULT_ADDR / VAULT_TOKEN / VAULT_SECRET_PATH not configured.")
        try:
            import hvac
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "hvac is required for the vault backend. Install it via `pip install hvac`."
            ) from exc

        client = hvac.Client(url=self._addr, token=self._token)
        read = client.secrets.kv.v2.read_secret_version(path=self._secret_path)
        self._cache = read["data"]["data"]
        return self._cache

    def get_secret(self, key: str) -> Optional[str]:
        try:
            return self._load().get(key)
        except SecretNotFoundError:
            return None


@functools.lru_cache(maxsize=1)
def get_secrets_provider() -> SecretsProvider:
    """Factory: returns the configured SecretsProvider singleton."""
    backend = os.environ.get("IDAMP_SECRETS_BACKEND", "env").lower()
    if backend == "aws_secrets_manager":
        return AWSSecretsManagerProvider()
    if backend == "vault":
        return VaultSecretsProvider()
    return EnvSecretsProvider()
