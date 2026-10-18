"""The host's cloud keys, kept in the operating system's keychain (macOS Keychain, Windows
Credential Manager) and never in the project folder. Guests' keys never come here: the server
keeps those in memory for their visit only."""

from __future__ import annotations

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

from upscaler.cloud.base import CloudError, clean_key
from upscaler.config import APP_NAME

SERVICE_NAME = APP_NAME  # keychain entries: service "Upscaler", username = provider id


def get_key(provider_id: str) -> str | None:
    try:
        return keyring.get_password(SERVICE_NAME, provider_id)
    except KeyringError:
        return None  # no usable keychain on this computer: behave as "not connected"


def set_key(provider_id: str, key: str) -> None:
    try:
        keyring.set_password(SERVICE_NAME, provider_id, clean_key(key))
    except KeyringError as exc:
        raise CloudError("Couldn't save the key in this computer's keychain. Please try again.") from exc


_NOT_REMOVED = "Couldn't remove the key from this computer's keychain. Please try again."


def delete_key(provider_id: str) -> None:
    try:
        keyring.delete_password(SERVICE_NAME, provider_id)
    except PasswordDeleteError as exc:
        # keyring raises this both for "there was no such key" and, on macOS, for any Keychain
        # error, such as someone clicking Deny or the keychain being locked. Only the first is fine.
        try:
            still_there = keyring.get_password(SERVICE_NAME, provider_id) is not None
        except KeyringError:
            still_there = True  # can't tell, so don't claim it's gone
        if still_there:
            raise CloudError(_NOT_REMOVED) from exc
    except KeyringError as exc:
        raise CloudError(_NOT_REMOVED) from exc
