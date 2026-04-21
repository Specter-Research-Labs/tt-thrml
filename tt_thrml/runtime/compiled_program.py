from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
import torch

from ..compiler.sampler_lowering import CompiledSamplerLowering
from ..runtime_config import ParameterKernelBackend
from ..tensor_specs import PhysicalTensorSpec


def node_kind_from_template(template_sd: jax.ShapeDtypeStruct) -> str:
    if template_sd.shape != ():
        raise TypeError(
            "TTProgramExecutor currently supports only scalar-array node states."
        )
    dtype_kind = np.dtype(template_sd.dtype).kind
    if dtype_kind == "b":
        return "spin"
    if dtype_kind in ("i", "u"):
        return "categorical"
    if dtype_kind == "f":
        return "continuous"
    raise TypeError(
        "TTProgramExecutor currently supports only scalar bool, integer, "
        f"or floating-point node states. Got {template_sd.dtype!r}."
    )


@dataclass(frozen=True)
class CompiledGlobalSlot:
    global_slot_index: int
    node_kind: str | None
    output_dtype: object | None
    device_dtype: object | None
    block_indices: tuple[int, ...]


@dataclass(frozen=True)
class CompiledStateView:
    block_index: int
    global_slot_index: int
    node_kind: str
    n_nodes: int
    output_dtype: object
    positions: np.ndarray
    gather_index: object
    host_gather_index: torch.Tensor
    tensor_spec: PhysicalTensorSpec


@dataclass(frozen=True)
class CompiledGatherShard:
    block_index: int
    repeat_sizes: tuple[int, int, int, int]
    gather_index: object
    membership_mask: object
    repeat_is_identity: bool
    gather_is_identity: bool
    membership_mask_is_all_ones: bool


@dataclass(frozen=True)
class CompiledInteractionSource:
    node_kind: str
    shards: tuple[CompiledGatherShard, ...]


@dataclass(frozen=True)
class CompiledDirectSourcePlan:
    block_index: int
    use_row_major: bool
    tensor_spec: PhysicalTensorSpec


@dataclass(frozen=True)
class CompiledGatherSourcePlan:
    shards: tuple[CompiledGatherShard, ...]
    tensor_spec: PhysicalTensorSpec


CompiledInteractionSourcePlan = CompiledDirectSourcePlan | CompiledGatherSourcePlan


@dataclass(frozen=True)
class CompiledInteractionExecution:
    spin_sources: tuple[CompiledInteractionSourcePlan, ...]
    categorical_sources: tuple[CompiledInteractionSourcePlan, ...]
    continuous_sources: tuple[CompiledInteractionSourcePlan, ...]


@dataclass(frozen=True)
class CompiledInteraction:
    contribution_kind: str
    n_interactions: int
    tail_shape: tuple[int, ...]
    categorical_tail_strides: tuple[int, ...]
    execution: CompiledInteractionExecution
    flat_weights: object
    active_mask: object
    active_mask_is_all_ones: bool
    parameter_spec: PhysicalTensorSpec
    flat_weights_spec: PhysicalTensorSpec
    active_mask_spec: PhysicalTensorSpec
    fused_static_theta_bias: bool
    use_single_node_fused_theta_scale_fast_path: bool
    fused_static_theta_prefix: object | None


@dataclass(frozen=True)
class CompiledInteractionGroup:
    interactions: tuple[CompiledInteraction, ...]
    n_interactions: int
    flat_weights: object | None
    active_mask: object | None
    flat_indices: object | None = None
    parameter_spec: PhysicalTensorSpec | None = None
    flat_weights_spec: PhysicalTensorSpec | None = None
    active_mask_spec: PhysicalTensorSpec | None = None


@dataclass(frozen=True)
class CompiledSpinFamilyRuntime:
    zero_parameters: object
    positive_ones: object
    negative_ones: object
    parameter_spec: PhysicalTensorSpec


@dataclass(frozen=True)
class CompiledCategoricalFamilyRuntime:
    zero_parameters: object
    static_bias: object | None
    sampling_plan: object | None
    parameter_spec: PhysicalTensorSpec


@dataclass(frozen=True)
class CompiledGaussianFamilyRuntime:
    zero_parameters: object
    linear_selector: object
    precision_selector: object
    parameter_spec: PhysicalTensorSpec


CompiledFamilyRuntime = (
    CompiledSpinFamilyRuntime
    | CompiledCategoricalFamilyRuntime
    | CompiledGaussianFamilyRuntime
)


@dataclass(frozen=True)
class CompiledBlock:
    block_index: int
    sampler_lowering: CompiledSamplerLowering
    parameter_kernel_backend: ParameterKernelBackend
    n_nodes: int
    output_dtype: object
    n_categories: int | None
    state_view: CompiledStateView
    interactions: tuple[CompiledInteraction | CompiledInteractionGroup, ...]
    family_runtime: CompiledFamilyRuntime


@dataclass(frozen=True)
class CompiledProgram:
    blocks: tuple[CompiledBlock, ...]
    state_views: tuple[CompiledStateView, ...]
    global_slots: tuple[CompiledGlobalSlot, ...]
    row_major_cache_block_indices: frozenset[int]
    state_layout: object
    categorical_layout: object
    spin_state_dtype: object
    categorical_state_dtype: object
    index_dtype: object


__all__ = [
    "CompiledBlock",
    "CompiledCategoricalFamilyRuntime",
    "CompiledDirectSourcePlan",
    "CompiledGatherShard",
    "CompiledGatherSourcePlan",
    "CompiledGlobalSlot",
    "CompiledGaussianFamilyRuntime",
    "CompiledInteraction",
    "CompiledInteractionExecution",
    "CompiledInteractionGroup",
    "CompiledInteractionSource",
    "CompiledProgram",
    "CompiledSpinFamilyRuntime",
    "CompiledStateView",
    "node_kind_from_template",
]
