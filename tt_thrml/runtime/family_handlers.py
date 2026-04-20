from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from jax import numpy as jnp
import jax

from ..compiler.categorical_ops import (
    CategoricalThetaInputs,
    categorical_flat_index_tensor_device,
    categorical_interaction_scale_tensor_device,
    categorical_tail_index_tensor_device,
)
from ..compiler.gaussian_ops import GaussianCanonicalInputs, gaussian_noise_tensor
from ..compiler.spin_ops import SpinGammaInputs
from ..runtime_config import (
    CATEGORICAL_PARAMETER_FAMILY,
    GAUSSIAN_PARAMETER_FAMILY,
    ParameterFamily,
    SPIN_PARAMETER_FAMILY,
)
from .compiled_program import (
    CompiledBlock,
    CompiledCategoricalFamilyRuntime,
    CompiledGaussianFamilyRuntime,
    CompiledInteraction,
    CompiledSpinFamilyRuntime,
)
from . import state_runtime as _state_runtime
from .runtime_utils import spin_threshold_logits_tensor


@dataclass(frozen=True)
class SpinPreparedRandom:
    threshold_logits: object


@dataclass(frozen=True)
class CategoricalPreparedRandom:
    gumbel_noise: object


@dataclass(frozen=True)
class GaussianPreparedRandom:
    gaussian_noise: object


PreparedFamilyRandom = (
    SpinPreparedRandom | CategoricalPreparedRandom | GaussianPreparedRandom | None
)


@dataclass(frozen=True)
class ParameterFamilyHandler:
    family: ParameterFamily
    compute_stage: str
    sample_stage: str
    compute_interaction_partial: Callable[..., object]
    initialize_parameters: Callable[..., object]
    sample: Callable[..., object]
    prepare_sample_inputs: Callable[..., tuple[object, ...]]
    prepare_batch_sample_inputs: Callable[..., object | None]
    parameters_to_host: Callable[..., object]


def _spin_runtime(block: CompiledBlock) -> CompiledSpinFamilyRuntime:
    runtime = block.family_runtime
    if not isinstance(runtime, CompiledSpinFamilyRuntime):
        raise TypeError(f"Block {block.block_index} is not a spin sampler block.")
    return runtime


def _categorical_runtime(block: CompiledBlock) -> CompiledCategoricalFamilyRuntime:
    runtime = block.family_runtime
    if not isinstance(runtime, CompiledCategoricalFamilyRuntime):
        raise TypeError(f"Block {block.block_index} is not a categorical sampler block.")
    return runtime


def _gaussian_runtime(block: CompiledBlock) -> CompiledGaussianFamilyRuntime:
    runtime = block.family_runtime
    if not isinstance(runtime, CompiledGaussianFamilyRuntime):
        raise TypeError(f"Block {block.block_index} is not a continuous Gaussian block.")
    return runtime


def _spin_threshold_tensor(executor, block: CompiledBlock, key):
    return executor.ttnn.from_torch(
        spin_threshold_logits_tensor(key, n_nodes=block.n_nodes),
        dtype=executor.compiled.spin_state_dtype,
        layout=executor.compiled.state_layout,
        device=executor.device,
    )


def _categorical_gumbel_tensor(executor, block: CompiledBlock, key):
    assert block.n_categories is not None
    return executor.ttnn.from_torch(
        torch.from_numpy(
            np.asarray(
                jax.random.gumbel(
                    key,
                    shape=(block.n_nodes, int(block.n_categories)),
                    dtype=jnp.float32,
                ),
                dtype=np.float32,
            ).copy()
        ).reshape(1, 1, block.n_nodes, int(block.n_categories)),
        dtype=executor.compiled.spin_state_dtype,
        layout=executor.compiled.categorical_layout,
        device=executor.device,
    )


def _gaussian_noise_device_tensor(executor, block: CompiledBlock, key):
    return executor.ttnn.from_torch(
        gaussian_noise_tensor(key, n_nodes=block.n_nodes),
        dtype=executor.compiled.spin_state_dtype,
        layout=executor.compiled.state_layout,
        device=executor.device,
    )


def _prepare_spin_sample_inputs(
    executor,
    block: CompiledBlock,
    sample_keys: Sequence[object],
) -> tuple[SpinPreparedRandom, ...]:
    return tuple(
        SpinPreparedRandom(
            threshold_logits=_spin_threshold_tensor(executor, block, sample_key)
        )
        for sample_key in sample_keys
    )


def _prepare_spin_batch_sample_inputs(
    executor,
    block: CompiledBlock,
    sample_keys: Sequence[object],
) -> SpinPreparedRandom:
    return SpinPreparedRandom(
        threshold_logits=executor.ttnn.from_torch(
            torch.concat(
                [
                    spin_threshold_logits_tensor(sample_key, n_nodes=block.n_nodes)
                    for sample_key in sample_keys
                ],
                dim=0,
            ),
            dtype=executor.compiled.spin_state_dtype,
            layout=executor.compiled.state_layout,
            device=executor.device,
        )
    )


def _prepare_categorical_sample_inputs(
    executor,
    block: CompiledBlock,
    sample_keys: Sequence[object],
) -> tuple[CategoricalPreparedRandom, ...]:
    return tuple(
        CategoricalPreparedRandom(
            gumbel_noise=_categorical_gumbel_tensor(executor, block, sample_key)
        )
        for sample_key in sample_keys
    )


def _prepare_gaussian_sample_inputs(
    executor,
    block: CompiledBlock,
    sample_keys: Sequence[object],
) -> tuple[GaussianPreparedRandom, ...]:
    return tuple(
        GaussianPreparedRandom(
            gaussian_noise=_gaussian_noise_device_tensor(executor, block, sample_key)
        )
        for sample_key in sample_keys
    )


def _unsupported_batch_sample_inputs(
    executor,
    block: CompiledBlock,
    sample_keys: Sequence[object],
) -> PreparedFamilyRandom:
    del executor, sample_keys
    raise TypeError(
        f"Batched sample inputs are not implemented for block {block.block_index} "
        f"({block.sampler_lowering.parameter_family.value})."
    )


def _compute_categorical_interaction_partial(
    executor,
    block: CompiledBlock,
    interaction: CompiledInteraction,
    spin_sources,
    categorical_sources,
    continuous_sources,
):
    def _build():
        if continuous_sources:
            raise TypeError(
                "categorical_logits interactions do not support continuous tails yet."
            )
        categorical_spin_sources = tuple(spin_sources)
        categorical_sources_row_major = tuple(categorical_sources)
        if interaction.fused_static_theta_bias:
            categorical_spin_sources = tuple(
                executor.ttnn.concat(
                    [interaction.fused_static_theta_prefix, source],
                    dim=1 if interaction.use_single_node_fused_theta_scale_fast_path else 2,
                )
                for source in categorical_spin_sources
            )
        flat_index = None
        if interaction.tail_shape:
            flat_index = categorical_flat_index_tensor_device(
                ttnn=executor.ttnn,
                device=executor.device,
                categorical_sources=categorical_sources_row_major,
                categorical_tail_strides=interaction.categorical_tail_strides,
                layout=executor.compiled.categorical_layout,
                index_dtype=executor.compiled.index_dtype,
            )

        interaction_scale = categorical_interaction_scale_tensor_device(
            ttnn=executor.ttnn,
            active_mask=interaction.active_mask,
            spin_sources=categorical_spin_sources,
            n_nodes=block.n_nodes,
            n_interactions=interaction.n_interactions,
            active_mask_is_all_ones=interaction.active_mask_is_all_ones,
            skip_flatten_if_aligned=interaction.use_single_node_fused_theta_scale_fast_path,
        )

        return executor._parameter_kernel_op_for_block(block)(
            ttnn=executor.ttnn,
            device=executor.device,
            inputs=executor._profile_call(
                "categorical_theta_inputs.build",
                lambda: CategoricalThetaInputs(
                    flat_weights=interaction.flat_weights,
                    flat_index=flat_index,
                    interaction_scale=interaction_scale,
                    n_nodes=block.n_nodes,
                    n_interactions=interaction.n_interactions,
                    n_categories=int(block.n_categories),
                ),
            ),
        )

    return _build()


def _compute_spin_interaction_partial(
    executor,
    block: CompiledBlock,
    interaction: CompiledInteraction,
    spin_sources,
    categorical_sources,
    continuous_sources,
):
    def _build():
        batch_size = int(executor._block_state_slots[block.state_view.block_index].shape[0])
        flat_index = None
        if interaction.tail_shape:
            flat_index = categorical_tail_index_tensor_device(
                ttnn=executor.ttnn,
                device=executor.device,
                categorical_sources=categorical_sources,
                categorical_tail_strides=interaction.categorical_tail_strides,
                n_nodes=block.n_nodes,
                n_interactions=interaction.n_interactions,
                n_categories=None,
                layout=executor.compiled.state_layout,
                index_dtype=executor.compiled.index_dtype,
            )

        interaction_scale = None
        if not interaction.active_mask_is_all_ones:
            interaction_scale = executor._ensure_tensor_batch_size(
                interaction.active_mask,
                batch_size,
            )

        source_scale = None
        for gathered_source in (*spin_sources, *continuous_sources):
            aligned_source = executor.ttnn.reshape(
                gathered_source,
                (batch_size, *interaction.parameter_scale_shape_tail),
            )
            source_scale = (
                aligned_source
                if source_scale is None
                else executor.ttnn.multiply(source_scale, aligned_source)
            )

        if source_scale is not None:
            interaction_scale = (
                source_scale
                if interaction_scale is None
                else executor.ttnn.multiply(interaction_scale, source_scale)
            )

        return executor._parameter_kernel_op_for_block(block)(
            ttnn=executor.ttnn,
            device=executor.device,
            inputs=executor._profile_call(
                "spin_gamma_inputs.build",
                lambda: SpinGammaInputs(
                    flat_weights=interaction.flat_weights,
                    flat_index=flat_index,
                    interaction_scale=interaction_scale,
                    n_nodes=block.n_nodes,
                    n_interactions=interaction.n_interactions,
                ),
            ),
        )

    return _build()


def _compute_gaussian_interaction_partial(
    executor,
    block: CompiledBlock,
    interaction: CompiledInteraction,
    spin_sources,
    categorical_sources,
    continuous_sources,
):
    def _build():
        batch_size = int(executor._block_state_slots[block.state_view.block_index].shape[0])
        flat_index = None
        if interaction.tail_shape:
            flat_index = categorical_tail_index_tensor_device(
                ttnn=executor.ttnn,
                device=executor.device,
                categorical_sources=categorical_sources,
                categorical_tail_strides=interaction.categorical_tail_strides,
                n_nodes=block.n_nodes,
                n_interactions=interaction.n_interactions,
                n_categories=None,
                layout=executor.compiled.state_layout,
                index_dtype=executor.compiled.index_dtype,
            )
        interaction_scale = None
        if not interaction.active_mask_is_all_ones:
            interaction_scale = executor._ensure_tensor_batch_size(
                interaction.active_mask,
                batch_size,
            )

        for source in (*spin_sources, *continuous_sources):
            aligned_source = executor.ttnn.reshape(
                source,
                (batch_size, *interaction.parameter_scale_shape_tail),
            )
            interaction_scale = (
                aligned_source
                if interaction_scale is None
                else executor.ttnn.multiply(interaction_scale, aligned_source)
            )

        return executor._parameter_kernel_op_for_block(block)(
            ttnn=executor.ttnn,
            device=executor.device,
            inputs=executor._profile_call(
                "gaussian_inputs.build",
                lambda: GaussianCanonicalInputs(
                    flat_weights=interaction.flat_weights,
                    flat_index=flat_index,
                    interaction_scale=interaction_scale,
                    n_nodes=block.n_nodes,
                    n_interactions=interaction.n_interactions,
                    contribution_kind=interaction.contribution_kind,
                ),
            ),
        )

    return _build()


def _initialize_spin_parameters(executor, block: CompiledBlock, batch_size: int):
    runtime = _spin_runtime(block)
    return executor._ensure_tensor_batch_size(runtime.zero_parameters, batch_size)


def _initialize_categorical_parameters(executor, block: CompiledBlock, batch_size: int):
    runtime = _categorical_runtime(block)
    base = runtime.static_bias
    if base is None:
        base = runtime.zero_parameters
    return executor._ensure_tensor_batch_size(base, batch_size)


def _initialize_gaussian_parameters(executor, block: CompiledBlock, batch_size: int):
    runtime = _gaussian_runtime(block)
    return executor._ensure_tensor_batch_size(runtime.zero_parameters, batch_size)


def _sample_spin_parameters(
    executor,
    block: CompiledBlock,
    key,
    parameters,
    prepared_random: PreparedFamilyRandom,
):
    runtime = _spin_runtime(block)
    batch_size = int(parameters.shape[0])
    threshold_logits = (
        prepared_random.threshold_logits
        if isinstance(prepared_random, SpinPreparedRandom)
        else None
    )
    if threshold_logits is None:
        threshold_logits = executor._profile_call(
            f"sample_block.block{block.block_index}.threshold_logits",
            lambda: _spin_threshold_tensor(executor, block, key),
        )
    threshold_logits = executor._ensure_tensor_batch_size(threshold_logits, batch_size)
    signed = executor._profile_call(
        f"sample_block.block{block.block_index}.device_op",
        lambda: executor.spin_sample_op(
            ttnn=executor.ttnn,
            device=executor.device,
            gamma=parameters,
            threshold_logits=threshold_logits,
            positive_ones=executor._ensure_tensor_batch_size(
                runtime.positive_ones, batch_size
            ),
            negative_ones=executor._ensure_tensor_batch_size(
                runtime.negative_ones, batch_size
            ),
        ),
    )
    return executor._profile_call(
        f"sample_block.block{block.block_index}.reshape",
        lambda: executor.ttnn.reshape(
            signed,
            (batch_size, 1, 1, block.n_nodes),
        ),
    )


def _sample_categorical_parameters(
    executor,
    block: CompiledBlock,
    key,
    parameters,
    prepared_random: PreparedFamilyRandom,
):
    runtime = _categorical_runtime(block)
    sampler_lowering = block.sampler_lowering
    categorical_sampler = executor.categorical_sampler
    if sampler_lowering.sample_categorical is not None:
        categorical_sampler = lambda **kwargs: sampler_lowering.sample_categorical(
            **kwargs,
            block=block,
            sampler_lowering=sampler_lowering,
            current_sampler_state=_state_runtime.sampler_state_for_block(
                executor, block.block_index
            ),
        )
    return executor._profile_call(
        f"sample_block.block{block.block_index}.categorical_sampler",
        lambda: categorical_sampler(
            ttnn=executor.ttnn,
            device=executor.device,
            logits=parameters,
            key=key,
            output_dtype=block.output_dtype,
            plan=runtime.sampling_plan,
            gumbel_noise=(
                prepared_random.gumbel_noise
                if isinstance(prepared_random, CategoricalPreparedRandom)
                else None
            ),
        ),
    )


def _sample_gaussian_parameters(
    executor,
    block: CompiledBlock,
    key,
    parameters,
    prepared_random: PreparedFamilyRandom,
):
    runtime = _gaussian_runtime(block)
    batch_size = int(parameters.shape[0])
    noise = (
        prepared_random.gaussian_noise
        if isinstance(prepared_random, GaussianPreparedRandom)
        else None
    )
    if noise is None:
        noise = executor._profile_call(
            f"sample_block.block{block.block_index}.gaussian_noise",
            lambda: _gaussian_noise_device_tensor(executor, block, key),
        )
    noise = executor._ensure_tensor_batch_size(noise, batch_size)
    linear_selector = executor._ensure_tensor_batch_size(
        runtime.linear_selector,
        batch_size,
    )
    precision_selector = executor._ensure_tensor_batch_size(
        runtime.precision_selector,
        batch_size,
    )
    linear = executor.ttnn.sum(
        executor.ttnn.multiply(parameters, linear_selector),
        dim=3,
        keepdim=True,
    )
    precision = executor.ttnn.sum(
        executor.ttnn.multiply(parameters, precision_selector),
        dim=3,
        keepdim=True,
    )
    variance = executor.ttnn.reciprocal(precision)
    mean = executor.ttnn.multiply(linear, variance)
    std = executor.ttnn.sqrt(variance)
    sample = executor.ttnn.add(mean, executor.ttnn.multiply(std, noise))
    return executor._profile_call(
        f"sample_block.block{block.block_index}.reshape",
        lambda: executor.ttnn.reshape(
            sample,
            (batch_size, 1, 1, block.n_nodes),
        ),
    )


def _spin_parameters_to_host(executor, block: CompiledBlock, parameters):
    del block
    return (0.5 * state_runtime.device_tensor_to_torch(executor, parameters)).reshape(-1)


def _categorical_parameters_to_host(executor, block: CompiledBlock, parameters):
    return state_runtime.device_tensor_to_torch(executor, parameters).reshape(
        block.n_nodes,
        int(block.n_categories),
    )


def _gaussian_parameters_to_host(executor, block: CompiledBlock, parameters):
    del block
    host_parameters = state_runtime.device_tensor_to_torch(executor, parameters).to(
        torch.float32
    )
    return (
        host_parameters[..., 0].reshape(-1),
        host_parameters[..., 1].reshape(-1),
    )


PARAMETER_FAMILY_HANDLERS: dict[ParameterFamily, ParameterFamilyHandler] = {
    SPIN_PARAMETER_FAMILY: ParameterFamilyHandler(
        family=SPIN_PARAMETER_FAMILY,
        compute_stage="compute_block_parameters.spin",
        sample_stage="sample_block.spin",
        compute_interaction_partial=_compute_spin_interaction_partial,
        initialize_parameters=_initialize_spin_parameters,
        sample=_sample_spin_parameters,
        prepare_sample_inputs=_prepare_spin_sample_inputs,
        prepare_batch_sample_inputs=_prepare_spin_batch_sample_inputs,
        parameters_to_host=_spin_parameters_to_host,
    ),
    CATEGORICAL_PARAMETER_FAMILY: ParameterFamilyHandler(
        family=CATEGORICAL_PARAMETER_FAMILY,
        compute_stage="compute_block_parameters.categorical",
        sample_stage="sample_block.categorical",
        compute_interaction_partial=_compute_categorical_interaction_partial,
        initialize_parameters=_initialize_categorical_parameters,
        sample=_sample_categorical_parameters,
        prepare_sample_inputs=_prepare_categorical_sample_inputs,
        prepare_batch_sample_inputs=_unsupported_batch_sample_inputs,
        parameters_to_host=_categorical_parameters_to_host,
    ),
    GAUSSIAN_PARAMETER_FAMILY: ParameterFamilyHandler(
        family=GAUSSIAN_PARAMETER_FAMILY,
        compute_stage="compute_block_parameters.gaussian",
        sample_stage="sample_block.gaussian",
        compute_interaction_partial=_compute_gaussian_interaction_partial,
        initialize_parameters=_initialize_gaussian_parameters,
        sample=_sample_gaussian_parameters,
        prepare_sample_inputs=_prepare_gaussian_sample_inputs,
        prepare_batch_sample_inputs=_unsupported_batch_sample_inputs,
        parameters_to_host=_gaussian_parameters_to_host,
    ),
}


__all__ = [
    "CategoricalPreparedRandom",
    "GaussianPreparedRandom",
    "PARAMETER_FAMILY_HANDLERS",
    "ParameterFamilyHandler",
    "PreparedFamilyRandom",
    "SpinPreparedRandom",
]
