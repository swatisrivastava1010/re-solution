"""Optional bring-your-own-key cloud upscalers."""

from __future__ import annotations

from upscaler.cloud.base import CloudError, CloudModel, CloudProvider
from upscaler.cloud.fal import FalProvider
from upscaler.cloud.magnific import MagnificProvider
from upscaler.cloud.replicate import ReplicateProvider

# Shown in this order in the app.
PROVIDERS: dict[str, CloudProvider] = {
    provider.id: provider for provider in (FalProvider(), ReplicateProvider(), MagnificProvider())
}


def get_provider(id: str) -> CloudProvider:
    """Raises KeyError for an unknown provider id."""
    return PROVIDERS[id]


__all__ = ["PROVIDERS", "CloudError", "CloudModel", "CloudProvider", "get_provider"]
