from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from thrml.block_management import Block
from thrml.block_sampling import BlockSamplingProgram
from thrml.models.discrete_ebm import CategoricalGibbsConditional, SpinGibbsConditional

from ..conditional_samplers import GaussianConditional
from ..runtime_config import (
    CATEGORICAL_PARAMETER_FAMILY,
    GAUSSIAN_PARAMETER_FAMILY,
    SPIN_PARAMETER_FAMILY,
    ParameterFamily,
    normalize_parameter_family,
    parameter_family_spec,
)


class StateViewLike(Protocol):
    node_kind: str
    n_nodes: int


@dataclass(frozen=True)
class SamplerStateConfig:
    shape: tuple[int, ...]
    output_dtype: object
    device_dtype: object | None = None
    layout_kind: str = "state"
    update_sampler_state: Callable[..., object] | None = None


@dataclass(frozen=True)
class SamplerLoweringConfig:
    parameter_family: ParameterFamily
    random_source_kind: str
    n_categories: int | None = None
    parameters_depend_on_sampler_state: bool = False
    sampler_state: SamplerStateConfig | None = None
    transform_parameters: Callable[..., object] | None = None
    sample_categorical: Callable[..., object] | None = None


@dataclass(frozen=True)
class CompiledSamplerStateSpec:
    shape: tuple[int, ...]
    output_dtype: object
    device_dtype: object
    layout: object
    update_sampler_state: Callable[..., object] | None


@dataclass(frozen=True)
class CompiledSamplerLowering:
    parameter_family: ParameterFamily
    random_source_kind: str
    n_categories: int | None
    parameters_depend_on_sampler_state: bool
    sampler_state_spec: CompiledSamplerStateSpec | None
    transform_parameters: Callable[..., object] | None
    sample_categorical: Callable[..., object] | None


def categorical_n_categories(node_or_type) -> int | None:
    targets = (
        (node_or_type, type(node_or_type))
        if not isinstance(node_or_type, type)
        else (node_or_type,)
    )
    for target in targets:
        for attr_name in ("n_categories", "num_categories", "cardinality"):
            value = getattr(target, attr_name, None)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{target!r}.{attr_name} must be an integer category count."
                )
            if int(value) < 2:
                raise ValueError(
                    f"{target!r}.{attr_name} must be at least 2 for categorical nodes."
                )
            return int(value)
    return None


def declared_parameter_family(sampler) -> ParameterFamily | None:
    family = getattr(sampler, "tt_parameter_family", None)
    if callable(family):
        family = family()
    if family is None:
        return None
    return normalize_parameter_family(family)


def sampler_kind(sampler) -> str | None:
    declared_family = declared_parameter_family(sampler)
    if declared_family is not None:
        return parameter_family_spec(declared_family).sampler_kind
    if isinstance(sampler, SpinGibbsConditional):
        return "spin"
    if isinstance(sampler, CategoricalGibbsConditional):
        return "categorical"
    if isinstance(sampler, GaussianConditional):
        return "continuous"
    return None


def program_supported_by_executor(program: BlockSamplingProgram) -> bool:
    return all(sampler_kind(sampler) is not None for sampler in program.samplers)


def unsupported_sampler_message(block_index: int, sampler) -> str:
    return (
        "TTProgramExecutor currently supports THRML discrete EBM Gibbs samplers, "
        "explicit TT sampler lowerings for spin/categorical/continuous samplers, and "
        "TT-MLIR-backed parameter kernels attached through the backend binding. "
        f"Found {type(sampler).__name__} at block {block_index}."
    )


def resolve_block_n_categories(block: Block, sampler) -> int:
    sampler_n_categories = getattr(sampler, "n_categories", None)
    node_n_categories = categorical_n_categories(block.node_type)

    if sampler_n_categories is not None:
        if isinstance(sampler_n_categories, bool) or not isinstance(
            sampler_n_categories, int
        ):
            raise TypeError(
                f"{type(sampler).__name__}.n_categories must be an integer."
            )
        sampler_n_categories = int(sampler_n_categories)
        if sampler_n_categories < 2:
            raise ValueError("Categorical samplers must declare at least 2 categories.")

    if (
        sampler_n_categories is not None
        and node_n_categories is not None
        and sampler_n_categories != node_n_categories
    ):
        raise ValueError(
            "Categorical block metadata mismatch: "
            f"{type(sampler).__name__}.n_categories={sampler_n_categories} "
            f"but {block.node_type.__name__}.n_categories={node_n_categories}."
        )

    if sampler_n_categories is not None:
        return sampler_n_categories
    if node_n_categories is not None:
        return node_n_categories

    raise TypeError(
        "Cannot determine categorical domain size for block "
        f"{block!r}. Declare sampler.n_categories or attach n_categories metadata "
        "to the categorical node type."
    )


def default_sampler_state_device_dtype(
    *,
    output_dtype,
    spin_state_dtype,
    index_dtype,
):
    dtype_kind = np.dtype(output_dtype).kind
    if dtype_kind in ("b", "i", "u"):
        return index_dtype
    if dtype_kind == "f":
        return spin_state_dtype
    raise TypeError(f"Unsupported sampler-state dtype: {output_dtype!r}.")


def resolve_sampler_lowering_config(
    *,
    sampler,
    block: Block,
    state_view: StateViewLike,
    ttnn,
    device,
    state_layout,
    categorical_layout,
    spin_state_dtype,
    categorical_state_dtype,
    index_dtype,
) -> SamplerLoweringConfig | None:
    def _lowering_config(
        parameter_family: ParameterFamily,
        *,
        n_categories: int | None = None,
    ) -> SamplerLoweringConfig:
        return SamplerLoweringConfig(
            parameter_family=parameter_family,
            random_source_kind=parameter_family_spec(parameter_family).random_source_kind,
            n_categories=n_categories,
        )

    lowering_factory = getattr(sampler, "tt_sampler_lowering_config", None)
    if callable(lowering_factory):
        lowering = lowering_factory(
            ttnn=ttnn,
            device=device,
            block=block,
            node_kind=state_view.node_kind,
            n_nodes=state_view.n_nodes,
            state_layout=state_layout,
            categorical_layout=categorical_layout,
            spin_state_dtype=spin_state_dtype,
            categorical_state_dtype=categorical_state_dtype,
            index_dtype=index_dtype,
        )
        if lowering is not None:
            if not isinstance(lowering, SamplerLoweringConfig):
                raise TypeError(
                    "tt_sampler_lowering_config() must return "
                    "SamplerLoweringConfig or None."
                )
            return lowering

    declared_family = declared_parameter_family(sampler)
    if declared_family == SPIN_PARAMETER_FAMILY:
        return _lowering_config(SPIN_PARAMETER_FAMILY)

    if declared_family == CATEGORICAL_PARAMETER_FAMILY:
        return _lowering_config(
            CATEGORICAL_PARAMETER_FAMILY,
            n_categories=resolve_block_n_categories(block, sampler),
        )

    if declared_family == GAUSSIAN_PARAMETER_FAMILY:
        return _lowering_config(GAUSSIAN_PARAMETER_FAMILY)

    if isinstance(sampler, SpinGibbsConditional):
        return _lowering_config(SPIN_PARAMETER_FAMILY)

    if isinstance(sampler, CategoricalGibbsConditional):
        return _lowering_config(
            CATEGORICAL_PARAMETER_FAMILY,
            n_categories=resolve_block_n_categories(block, sampler),
        )

    if isinstance(sampler, GaussianConditional):
        return _lowering_config(GAUSSIAN_PARAMETER_FAMILY)

    return None


def compile_sampler_state_spec(
    state_config: SamplerStateConfig,
    *,
    state_layout,
    categorical_layout,
    spin_state_dtype,
    index_dtype,
) -> CompiledSamplerStateSpec:
    shape = tuple(int(size) for size in state_config.shape)
    if any(size <= 0 for size in shape):
        raise ValueError("Sampler-state shapes must contain only positive sizes.")

    if state_config.layout_kind == "state":
        layout = state_layout
    elif state_config.layout_kind == "categorical":
        layout = categorical_layout
    else:
        raise ValueError(
            "Sampler-state layout_kind must be 'state' or 'categorical'."
        )

    output_dtype = np.dtype(state_config.output_dtype)
    device_dtype = state_config.device_dtype
    if device_dtype is None:
        device_dtype = default_sampler_state_device_dtype(
            output_dtype=output_dtype,
            spin_state_dtype=spin_state_dtype,
            index_dtype=index_dtype,
        )

    return CompiledSamplerStateSpec(
        shape=shape,
        output_dtype=output_dtype,
        device_dtype=device_dtype,
        layout=layout,
        update_sampler_state=state_config.update_sampler_state,
    )


def compile_sampler_lowering(
    lowering_config: SamplerLoweringConfig,
    *,
    sampler,
    block: Block,
    state_view: StateViewLike,
    state_layout,
    categorical_layout,
    spin_state_dtype,
    index_dtype,
) -> CompiledSamplerLowering:
    parameter_family = normalize_parameter_family(lowering_config.parameter_family)
    family_spec = parameter_family_spec(parameter_family)
    if state_view.node_kind != family_spec.output_node_kind:
        raise TypeError(
            f"{parameter_family.value} sampler lowering requires a "
            f"{family_spec.output_node_kind} output block."
        )

    if parameter_family == SPIN_PARAMETER_FAMILY:
        if lowering_config.sample_categorical is not None:
            raise TypeError(
                "sample_categorical sampler lowering is only supported for "
                "categorical_logits parameter families."
            )
        n_categories = None
    elif parameter_family == CATEGORICAL_PARAMETER_FAMILY:
        n_categories = lowering_config.n_categories
        if n_categories is None:
            n_categories = resolve_block_n_categories(block, sampler)
        if isinstance(n_categories, bool) or not isinstance(n_categories, int):
            raise TypeError("Categorical sampler lowering n_categories must be an integer.")
        n_categories = int(n_categories)
        if n_categories < 2:
            raise ValueError("Categorical sampler lowering requires at least 2 categories.")
    elif parameter_family == GAUSSIAN_PARAMETER_FAMILY:
        if lowering_config.sample_categorical is not None:
            raise TypeError(
                "sample_categorical sampler lowering is only supported for "
                "categorical_logits parameter families."
            )
        n_categories = None
    else:
        raise TypeError(f"Unsupported parameter family: {parameter_family!r}.")

    if (
        lowering_config.parameters_depend_on_sampler_state
        and lowering_config.sampler_state is None
    ):
        raise ValueError(
            "parameters_depend_on_sampler_state requires a sampler-state spec."
        )

    sampler_state_spec = (
        None
        if lowering_config.sampler_state is None
        else compile_sampler_state_spec(
            lowering_config.sampler_state,
            state_layout=state_layout,
            categorical_layout=categorical_layout,
            spin_state_dtype=spin_state_dtype,
            index_dtype=index_dtype,
        )
    )

    return CompiledSamplerLowering(
        parameter_family=parameter_family,
        random_source_kind=lowering_config.random_source_kind,
        n_categories=n_categories,
        parameters_depend_on_sampler_state=bool(
            lowering_config.parameters_depend_on_sampler_state
        ),
        sampler_state_spec=sampler_state_spec,
        transform_parameters=lowering_config.transform_parameters,
        sample_categorical=lowering_config.sample_categorical,
    )


__all__ = [
    "CompiledSamplerLowering",
    "CompiledSamplerStateSpec",
    "SamplerLoweringConfig",
    "SamplerStateConfig",
    "compile_sampler_lowering",
    "compile_sampler_state_spec",
    "declared_parameter_family",
    "default_sampler_state_device_dtype",
    "sampler_kind",
    "program_supported_by_executor",
    "resolve_block_n_categories",
    "resolve_sampler_lowering_config",
    "unsupported_sampler_message",
]
