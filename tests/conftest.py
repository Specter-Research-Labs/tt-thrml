"""Test-only metadata compatibility shims."""

from __future__ import annotations

import importlib.metadata


_original_version = importlib.metadata.version


def _version_with_local_fallback(distribution_name: str) -> str:
    try:
        return _original_version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        if distribution_name in {"thrml", "tt-thrml"}:
            return "0+local"
        raise


importlib.metadata.version = _version_with_local_fallback
