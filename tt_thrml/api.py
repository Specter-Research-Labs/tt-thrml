"""Primary TT-Lang public API."""

from __future__ import annotations

from typing import Any


def make_executor(
    ttnn: Any,
    device: Any,
    program: Any,
    config: Any | None = None,
    *,
    n_sweeps: int = 100,
    profile: bool = False,
) -> Any:
    """Build the primary TT-Lang executor for supported THRML program shapes."""
    if config is not None:
        raise ValueError("the primary TT-Lang executor does not use TT-MLIR config")
    if n_sweeps != 100:
        raise ValueError("the primary TT-Lang executor does not use n_sweeps; configure sweeps when running")
    if profile:
        raise ValueError("the primary TT-Lang executor does not use TT-MLIR profiling")

    from .ttlang_runtime import make_primary_ttlang_executor

    return make_primary_ttlang_executor(ttnn, device, program)
