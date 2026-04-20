"""Shared helpers for small tt-thrml hardware demos."""

from __future__ import annotations

import os
from pathlib import Path

import tt_thrml


def resolve_system_desc_path(system_desc_path: Path | None) -> Path | None:
    if system_desc_path is not None:
        return system_desc_path
    env_value = os.environ.get("SYSTEM_DESC_PATH")
    if not env_value:
        return None
    return Path(env_value)


def default_artifact_root(demo_name: str) -> Path:
    specter_artifact_root = os.environ.get("SPECTER_ARTIFACT_ROOT")
    if specter_artifact_root:
        return Path(specter_artifact_root) / "tt-thrml" / "demo-runs" / demo_name
    return Path("/tmp/tt-thrml-artifacts") / demo_name


def make_backend(
    *,
    ttnn_module,
    device,
    parameter_kernels: str,
    system_desc_path: Path | None,
    artifact_root: Path,
):
    backend = tt_thrml.make_backend_binding(ttnn_module, device)
    if parameter_kernels == "native":
        return backend
    resolved_system_desc = resolve_system_desc_path(system_desc_path)
    if resolved_system_desc is None:
        raise ValueError(
            "TT-MLIR demos require --system-desc-path or SYSTEM_DESC_PATH."
        )
    build_dir = os.environ.get("TTMLIR_BUILD_DIR")
    return tt_thrml.make_ttmlir_backend_binding(
        ttnn_module,
        device,
        system_desc_path=resolved_system_desc,
        artifact_root=artifact_root,
        build_dir=build_dir,
    )
