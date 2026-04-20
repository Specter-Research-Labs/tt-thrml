from __future__ import annotations

from pathlib import Path

from ...runtime_config import (
    CATEGORICAL_PARAMETER_FAMILY,
    GAUSSIAN_PARAMETER_FAMILY,
    BackendBinding,
    ParameterFamily,
    ParameterKernelBackend,
    SPIN_PARAMETER_FAMILY,
    make_backend_binding,
)
from .runtime import TTMLIRConfig, make_ttmlir_config


def make_ttmlir_parameter_kernel_ops(
    *,
    config: TTMLIRConfig | None = None,
    system_desc_path: Path | str | None = None,
    artifact_root: Path | str | None = None,
    build_dir: Path | str | None = None,
    ttmlir_opt: Path | str | None = None,
    ttmlir_translate: Path | str | None = None,
) -> dict[ParameterFamily, object]:
    from .categorical_theta import make_ttmlir_categorical_theta_op
    from .gaussian_canonical import make_ttmlir_gaussian_canonical_op
    from .spin_gamma import make_ttmlir_spin_gamma_op

    resolved_config = make_ttmlir_config(
        config=config,
        system_desc_path=system_desc_path,
        artifact_root=artifact_root,
        build_dir=build_dir,
        ttmlir_opt=ttmlir_opt,
        ttmlir_translate=ttmlir_translate,
    )
    shared_kwargs = {"config": resolved_config}
    return {
        CATEGORICAL_PARAMETER_FAMILY: make_ttmlir_categorical_theta_op(**shared_kwargs),
        GAUSSIAN_PARAMETER_FAMILY: make_ttmlir_gaussian_canonical_op(**shared_kwargs),
        SPIN_PARAMETER_FAMILY: make_ttmlir_spin_gamma_op(**shared_kwargs),
    }


def make_ttmlir_parameter_kernel_backends() -> dict[ParameterFamily, ParameterKernelBackend]:
    return {
        CATEGORICAL_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
        GAUSSIAN_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
        SPIN_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
    }


def make_ttmlir_backend_binding(
    ttnn,
    device,
    *,
    config: TTMLIRConfig | None = None,
    system_desc_path: Path | str | None = None,
    artifact_root: Path | str | None = None,
    build_dir: Path | str | None = None,
    ttmlir_opt: Path | str | None = None,
    ttmlir_translate: Path | str | None = None,
) -> BackendBinding:
    resolved_config = make_ttmlir_config(
        config=config,
        system_desc_path=system_desc_path,
        artifact_root=artifact_root,
        build_dir=build_dir,
        ttmlir_opt=ttmlir_opt,
        ttmlir_translate=ttmlir_translate,
    )
    return make_backend_binding(
        ttnn,
        device,
        parameter_kernel_backends=make_ttmlir_parameter_kernel_backends(),
        parameter_kernel_ops=make_ttmlir_parameter_kernel_ops(config=resolved_config),
    )


__all__ = [
    "TTMLIRConfig",
    "make_ttmlir_backend_binding",
    "make_ttmlir_config",
    "make_ttmlir_parameter_kernel_backends",
    "make_ttmlir_parameter_kernel_ops",
]
