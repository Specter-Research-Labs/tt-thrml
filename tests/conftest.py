"""Test-only shims: torch stub and metadata compatibility."""

from __future__ import annotations

import importlib.metadata

from tests.parity._torch_stub import install_torch_stub

install_torch_stub()


_original_version = importlib.metadata.version


def _version_with_local_fallback(distribution_name: str) -> str:
    try:
        return _original_version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        if distribution_name in {"thrml", "tt-thrml"}:
            return "0+local"
        raise


importlib.metadata.version = _version_with_local_fallback
