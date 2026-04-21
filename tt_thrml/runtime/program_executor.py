from __future__ import annotations

import time
from typing import Sequence

import jax
import numpy as np

from thrml.block_management import Block
from thrml.block_sampling import (
    BlockSamplingProgram,
    SamplingSchedule,
)
from thrml.observers import AbstractObserver, MomentAccumulatorObserver

from ..compiler.categorical_ops import (
    CategoricalSampler,
    dense_categorical_theta_op,
    exact_ttnn_categorical_sampler,
)
from ..compiler.device_contract import raise_host_fallback_disabled
from ..compiler.gaussian_ops import (
    dense_gaussian_canonical_op,
)
from ..compiler.program_compiler import compile_program
from ..runtime_config import (
    CATEGORICAL_PARAMETER_FAMILY,
    GAUSSIAN_PARAMETER_FAMILY,
    ParameterKernelBackend,
    SPIN_PARAMETER_FAMILY,
    merge_parameter_kernel_ops,
)
from .compiled_program import (
    CompiledBlock,
    CompiledDirectSourcePlan,
    CompiledGatherSourcePlan,
    CompiledInteraction,
    CompiledInteractionGroup,
    CompiledProgram,
)
from .family_handlers import (
    PARAMETER_FAMILY_HANDLERS,
    PreparedFamilyRandom,
)
from . import observation_runtime as _observation_runtime
from . import state_runtime as _state_runtime
from .mesh_support import canonical_replica_to_torch, is_multi_device_mesh
from .execution_support import (
    CompiledMomentObserverPlan,
    CompiledObservationPlan,
    LoadedObservationJob,
    LoadedStateJob,
    RuntimeProfile,
    PreparedRunRandoms,
    PreparedScheduleRandoms,
)
from ..compiler.spin_ops import dense_spin_gamma_op


def _emit_progress(progress, message: str) -> None:
    if progress is None:
        return
    progress(message)


def _sample_keys_for_iteration_keys(iteration_keys: Sequence[object], *, n_blocks: int):
    return tuple(
        tuple(
            jax.random.split(block_key, 2)[0]
            for block_key in jax.random.split(iter_key, n_blocks)
        )
        for iter_key in iteration_keys
    )


def _ensure_index_tensor_dtype(ttnn, device, value, *, dtype):
    current_dtype = getattr(value, "dtype", None)
    if current_dtype == dtype:
        return value

    typecast = getattr(ttnn, "typecast", None)
    if callable(typecast):
        value = typecast(value, dtype=dtype)
    else:
        to_dtype = getattr(ttnn, "to_dtype", None)
        if callable(to_dtype):
            value = to_dtype(value, dtype)

    current_dtype = getattr(value, "dtype", None)
    if current_dtype == dtype:
        return value
    raise_host_fallback_disabled(
        "index tensor dtype coercion",
        remedy=(
            "Use a TT backend that can cast gather indices on-device, or "
            "compile index tensors directly in the required dtype."
        ),
    )


def _default_spin_sample_op(
    *,
    ttnn,
    device,
    gamma,
    threshold_logits,
    positive_ones,
    negative_ones,
):
    del device
    gt = getattr(ttnn, "gt", None)
    where = getattr(ttnn, "where", None)
    if not callable(gt) or not callable(where):
        raise TypeError("TT backend must expose gt() and where() for TTNN spin sampling.")
    return where(
        gt(gamma, threshold_logits),
        positive_ones,
        negative_ones,
    )


class TTProgramExecutor:
    """Cached TT executor with device-resident block state.

    This executor keeps canonical per-block state on device, rebuilds THRML's
    grouped global slots on device with ``concat``, computes per-block ``gamma``
    and ``theta`` from that cached state, and updates free blocks without
    host-side patching in the hot loop.
    """

    def __init__(
        self,
        *,
        ttnn,
        device,
        program: BlockSamplingProgram,
        compiled: CompiledProgram | None = None,
        parameter_kernel_ops=None,
        parameter_kernel_backends=None,
        spin_sample_op=None,
        categorical_sampler: CategoricalSampler | None = None,
        profiler: RuntimeProfile | None = None,
        profile_sync: bool = False,
        progress=None,
    ):
        self.ttnn = ttnn
        self.device = device
        self.program = program
        self._parameter_family_ops = merge_parameter_kernel_ops(parameter_kernel_ops)
        self._parameter_kernel_backends = dict(parameter_kernel_backends or {})
        self._reference_parameter_family_ops = {
            SPIN_PARAMETER_FAMILY: dense_spin_gamma_op,
            CATEGORICAL_PARAMETER_FAMILY: dense_categorical_theta_op,
            GAUSSIAN_PARAMETER_FAMILY: dense_gaussian_canonical_op,
        }
        self.spin_sample_op = spin_sample_op or _default_spin_sample_op
        self.categorical_sampler = (
            categorical_sampler or exact_ttnn_categorical_sampler
        )
        self.profiler = profiler
        self.profile_sync = bool(profile_sync)
        self.progress = progress
        self.compiled = compiled if compiled is not None else self._compile_program(program)
        self._validate_parameter_kernel_plan()
        self._block_state_slots: list[object | None] = []
        self._block_state_slots_row_major: list[object | None] = []
        self._global_state_slots: list[object | None] = []
        self._sampler_states: list[object] | None = None
        self._sampler_state_slots: list[object | None] = []
        self._observation_plan_cache: dict[tuple[int, ...], CompiledObservationPlan] = {}
        self._moment_observer_plan_cache: dict[int, CompiledMomentObserverPlan] = {}
        self._expanded_batch_tensor_cache: dict[tuple[int, int], object] = {}
        self._shape_dtypes_cache = {
            node_type: jax.tree.unflatten(*sd)
            for node_type, sd in self.program.gibbs_spec.node_shape_dtypes.items()
        }
        self._program_block_id_to_index = {
            id(block): block_index
            for block_index, block in enumerate(program.gibbs_spec.blocks)
        }
        self._enable_grouped_moment_clamped_numpy_cache = True
        self._enable_global_source_slot_cache = True

    def _maybe_profile_sync(self) -> None:
        if not self.profile_sync:
            return
        synchronize_device = getattr(self.ttnn, "synchronize_device", None)
        if callable(synchronize_device):
            synchronize_device(self.device)

    def _profile_call(self, stage: str, fn):
        if self.profiler is None and self.progress is None and not self.profile_sync:
            return fn()

        _emit_progress(self.progress, f"[tt-executor] {stage}:start")
        start = time.perf_counter()
        result = fn()
        self._maybe_profile_sync()
        seconds = time.perf_counter() - start
        if self.profiler is not None:
            self.profiler.record(stage, seconds)
        _emit_progress(
            self.progress,
            f"[tt-executor] {stage}:done seconds={seconds:.6f}",
        )
        return result

    def _compile_program(self, program: BlockSamplingProgram) -> CompiledProgram:
        return compile_program(
            ttnn=self.ttnn,
            device=self.device,
            program=program,
            parameter_kernel_backends=self._parameter_kernel_backends,
        )

    def _validate_parameter_kernel_plan(self) -> None:
        missing = sorted(
            {
                (
                    block.sampler_lowering.parameter_family.value,
                    block.parameter_kernel_backend.value,
                )
                for block in self.compiled.blocks
                if block.parameter_kernel_backend is not ParameterKernelBackend.NATIVE
                and block.sampler_lowering.parameter_family not in self._parameter_family_ops
            }
        )
        if not missing:
            return
        raise TypeError(
            "Compiled program requires explicit parameter kernel ops for: "
            f"{missing!r}."
        )

    def _parameter_kernel_op_for_block(self, block: CompiledBlock):
        parameter_family = block.sampler_lowering.parameter_family
        if block.parameter_kernel_backend is ParameterKernelBackend.NATIVE:
            return self._reference_parameter_family_ops[parameter_family]

        parameter_kernel_op = self._parameter_family_ops.get(parameter_family)
        if parameter_kernel_op is None:
            raise TypeError(
                "Missing configured parameter kernel op for "
                f"{parameter_family.value} with backend "
                f"{block.parameter_kernel_backend.value!r}."
            )
        return parameter_kernel_op

    def _ensure_tensor_batch_size(self, value, batch_size: int):
        return _state_runtime.ensure_tensor_batch_size(self, value, batch_size)

    def set_sampler_states(self, sampler_states: Sequence[object]) -> None:
        _state_runtime.set_sampler_states(self, sampler_states)

    def copy_sampler_states(self) -> list[object]:
        return _state_runtime.copy_sampler_states(self)

    def load_free_state(self, state_free) -> None:
        _state_runtime.load_free_state(self, state_free)

    def load_free_state_batch(self, state_free_batch: Sequence[object]) -> None:
        _state_runtime.load_free_state_batch(self, state_free_batch)

    def load_clamp_state(self, state_clamp) -> None:
        _state_runtime.load_clamp_state(self, state_clamp)

    def load_clamp_state_batch(self, state_clamp, *, batch_size: int) -> None:
        _state_runtime.load_clamp_state_batch(self, state_clamp, batch_size=batch_size)

    def load_state(self, state_free, state_clamp) -> None:
        _state_runtime.load_state(self, state_free, state_clamp)

    @property
    def state_is_loaded(self) -> bool:
        return _state_runtime.state_is_loaded(self)

    def _require_state(self) -> None:
        _state_runtime.require_state(self)

    def _prepare_run_randoms(self, iteration_keys: Sequence[object]) -> PreparedRunRandoms:
        n_iters = len(iteration_keys)
        if n_iters == 0:
            return PreparedRunRandoms(
                iteration_keys=tuple(),
                sample_keys=tuple(),
                prepared_randoms_by_block={},
            )

        n_blocks = len(self.program.gibbs_spec.free_blocks)
        sample_keys = _sample_keys_for_iteration_keys(iteration_keys, n_blocks=n_blocks)

        prepared_randoms_by_block: dict[int, object] = {}

        for block in self.compiled.blocks:
            handler = self._parameter_family_handler(block)
            block_sample_keys = tuple(
                sample_keys[iter_index][block.block_index] for iter_index in range(n_iters)
            )
            prepared_randoms_by_block[block.block_index] = handler.prepare_batch_sample_inputs(
                self,
                block,
                block_sample_keys,
            )

        return PreparedRunRandoms(
            iteration_keys=tuple(iteration_keys),
            sample_keys=sample_keys,
            prepared_randoms_by_block=prepared_randoms_by_block,
        )

    def _prepare_schedule_randoms(
        self, key, schedule: SamplingSchedule
    ) -> PreparedScheduleRandoms:
        key, warmup_root = jax.random.split(key, 2)
        warmup_iteration_keys = tuple(jax.random.split(warmup_root, schedule.n_warmup))
        sample_keys = (
            tuple(jax.random.split(key, schedule.n_samples - 1))
            if schedule.n_samples > 1
            else tuple()
        )
        sample_iteration_groups = tuple(
            tuple(jax.random.split(sample_key, schedule.steps_per_sample))
            for sample_key in sample_keys
        )
        flattened_iteration_keys = warmup_iteration_keys + tuple(
            iteration_key
            for group in sample_iteration_groups
            for iteration_key in group
        )
        run_randoms = self._prepare_run_randoms(flattened_iteration_keys)
        warmup_count = len(warmup_iteration_keys)
        return PreparedScheduleRandoms(
            run_randoms=run_randoms,
            warmup_count=warmup_count,
            steps_per_sample=schedule.steps_per_sample,
            sample_interval_offsets=tuple(
                warmup_count + group_index * schedule.steps_per_sample
                for group_index in range(len(sample_iteration_groups))
            ),
        )

    def current_state_free_host(self):
        return _observation_runtime.current_state_free_host(self)

    def current_state_clamp_host(self):
        return _observation_runtime.current_state_clamp_host(self)

    def compile_observation_plan(
        self, blocks: Sequence[Block]
    ) -> CompiledObservationPlan:
        return _observation_runtime.compile_observation_plan(self, blocks)

    def observe_compiled_plan_host(self, plan: CompiledObservationPlan):
        return _observation_runtime.observe_compiled_plan_host(self, plan)

    def observe_compiled_plan_numpy(self, plan: CompiledObservationPlan):
        return _observation_runtime.observe_compiled_plan_numpy(self, plan)

    def observe_compiled_plan_numpy_batch(self, plan: CompiledObservationPlan):
        return _observation_runtime.observe_compiled_plan_numpy_batch(self, plan)

    def observe_compiled_plan_numpy_batch_moment_helper(
        self,
        plan: CompiledObservationPlan,
        *,
        clamped_block_numpy_cache: dict[int, np.ndarray] | None = None,
    ):
        return _observation_runtime.observe_compiled_plan_numpy_batch_moment_helper(
            self,
            plan,
            clamped_block_numpy_cache=clamped_block_numpy_cache,
        )

    def observe_blocks_host(self, blocks: Sequence[Block]):
        return _observation_runtime.observe_blocks_host(self, blocks)

    def compile_moment_observer_plan(
        self, observer: MomentAccumulatorObserver
    ) -> CompiledMomentObserverPlan:
        return _observation_runtime.compile_moment_observer_plan(self, observer)

    def _gather_source_plan(self, source_plan, *, batch_size: int):
        if isinstance(source_plan, CompiledDirectSourcePlan):
            gathered_state = (
                _state_runtime.row_major_block_state(self, source_plan.block_index)
                if source_plan.use_row_major
                else self._block_state_slots[source_plan.block_index]
            )
        elif isinstance(source_plan, CompiledGatherSourcePlan):
            gathered_state = None
            for shard in source_plan.shards:
                source_state = self._block_state_slots[shard.block_index]
                repeated_state = (
                    source_state
                    if shard.repeat_is_identity
                    else self.ttnn.repeat(source_state, shard.repeat_sizes)
                )
                gather_index = self._ensure_tensor_batch_size(
                    shard.gather_index, batch_size
                )
                gather_index = _ensure_index_tensor_dtype(
                    self.ttnn,
                    self.device,
                    gather_index,
                    dtype=self.compiled.index_dtype,
                )
                fragment = (
                    repeated_state
                    if shard.gather_is_identity
                    else self.ttnn.gather(repeated_state, -1, index=gather_index)
                )
                membership_mask = self._ensure_tensor_batch_size(
                    shard.membership_mask, batch_size
                )
                fragment = (
                    fragment
                    if shard.membership_mask_is_all_ones
                    else self.ttnn.multiply(fragment, membership_mask)
                )
                gathered_state = (
                    fragment
                    if gathered_state is None
                    else self.ttnn.add(gathered_state, fragment)
                )
        else:
            raise TypeError(f"Unsupported compiled source plan: {type(source_plan)!r}")

        if gathered_state is None:
            raise RuntimeError("Compiled interaction source plan produced no state.")

        target_shape = source_plan.tensor_spec.shape(batch_size)
        reshaped_source = (
            gathered_state
            if tuple(gathered_state.shape) == target_shape
            else self.ttnn.reshape(gathered_state, target_shape)
        )
        if source_plan.tensor_spec.layout is not None:
            return _state_runtime.maybe_to_layout(
                self, reshaped_source, source_plan.tensor_spec.layout
            )
        return reshaped_source

    def _gather_interaction_sources(
        self, block: CompiledBlock, interaction: CompiledInteraction
    ):
        def _gather():
            batch_size = int(
                self._block_state_slots[block.state_view.block_index].shape[0]
            )
            execution = interaction.execution
            return (
                tuple(
                    self._gather_source_plan(source_plan, batch_size=batch_size)
                    for source_plan in execution.spin_sources
                ),
                tuple(
                    self._gather_source_plan(source_plan, batch_size=batch_size)
                    for source_plan in execution.categorical_sources
                ),
                tuple(
                    self._gather_source_plan(source_plan, batch_size=batch_size)
                    for source_plan in execution.continuous_sources
                ),
            )

        return self._profile_call("interaction_sources.gather", _gather)

    def _parameter_family_handler(self, block: CompiledBlock):
        return PARAMETER_FAMILY_HANDLERS[block.sampler_lowering.parameter_family]

    def compute_block_parameters(self, block_index: int):
        block = self.compiled.blocks[block_index]
        handler = self._parameter_family_handler(block)

        def _compute():
            self._require_state()
            batch_size = int(
                self._block_state_slots[block.state_view.block_index].shape[0]
            )
            parameters = handler.initialize_parameters(
                self,
                block,
                batch_size,
            )
            for interaction in block.interactions:
                if isinstance(interaction, CompiledInteractionGroup):
                    partial = handler.compute_interaction_group_partial(
                        self,
                        block,
                        interaction,
                    )
                else:
                    spin_sources, categorical_sources, continuous_sources = (
                        self._gather_interaction_sources(
                            block,
                            interaction,
                        )
                    )
                    partial = handler.compute_interaction_partial(
                        self,
                        block,
                        interaction,
                        spin_sources,
                        categorical_sources,
                        continuous_sources,
                    )
                parameters = self.ttnn.add(parameters, partial)
            return self._transform_sampler_parameters(
                block=block,
                parameters=parameters,
            )

        return self._profile_call(handler.compute_stage, _compute)

    def compute_block_parameters_host(self, block_index: int):
        block = self.compiled.blocks[block_index]
        parameters = self.compute_block_parameters(block_index)
        return self._parameter_family_handler(block).parameters_to_host(
            self,
            block,
            parameters,
        )

    def _sample_is_device_tensor(self, sample) -> bool:
        return _state_runtime.sample_is_device_tensor(self, sample)

    def _coerce_sample_to_block_state_tensor(
        self, block: CompiledBlock, sample, *, layout
    ):
        return _state_runtime.coerce_sample_to_block_state_tensor(
            self,
            block,
            sample,
            layout=layout,
        )

    def _write_block_state(self, block_index: int, new_state) -> int:
        return _state_runtime.write_block_state(self, block_index, new_state)

    def _update_sampler_state(
        self,
        *,
        block: CompiledBlock,
        sample_key,
        sampler_state,
        sample,
    ):
        sampler_lowering = block.sampler_lowering
        sampler_state_spec = sampler_lowering.sampler_state_spec
        if (
            sampler_state_spec is None
            or sampler_state_spec.update_sampler_state is None
        ):
            return sampler_state
        return self._profile_call(
            f"sample_single_block.block{block.block_index}.sampler_state",
            lambda: sampler_state_spec.update_sampler_state(
                ttnn=self.ttnn,
                device=self.device,
                block=block,
                sampler_lowering=sampler_lowering,
                current_sampler_state=sampler_state,
                sample=sample,
                sample_key=sample_key,
            ),
        )

    def _transform_sampler_parameters(
        self,
        *,
        block: CompiledBlock,
        parameters,
    ):
        sampler_lowering = block.sampler_lowering
        if not sampler_lowering.parameters_depend_on_sampler_state:
            return parameters
        if sampler_lowering.transform_parameters is None:
            raise NotImplementedError(
                "Sampler-state-dependent parameter lowering is not "
                "implemented for this sampler."
            )
        sampler_state = _state_runtime.sampler_state_for_block(self, block.block_index)
        return self._profile_call(
            f"compute_block_parameters.block{block.block_index}.sampler_state",
            lambda: sampler_lowering.transform_parameters(
                ttnn=self.ttnn,
                device=self.device,
                block=block,
                sampler_lowering=sampler_lowering,
                current_sampler_state=sampler_state,
                parameters=parameters,
            ),
        )

    def _sample_block(
        self,
        key,
        block_index: int,
        *,
        prepared_random: PreparedFamilyRandom = None,
    ):
        block = self.compiled.blocks[block_index]
        handler = self._parameter_family_handler(block)

        def _sample():
            parameters = self._profile_call(
                f"sample_block.block{block_index}.parameters",
                lambda: self.compute_block_parameters(block_index),
            )
            return handler.sample(
                self,
                block,
                key,
                parameters,
                prepared_random,
            )

        return self._profile_call(
            handler.sample_stage,
            lambda: self._profile_call(
                f"sample_block.block{block_index}",
                _sample,
            ),
        )

    def _sample_single_block_from_sample_key(
        self,
        sample_key,
        block_index: int,
        *,
        sampler_state,
        prepared_random: PreparedFamilyRandom = None,
    ):
        def _sample():
            block = self.compiled.blocks[block_index]
            sample = self._sample_block(
                sample_key,
                block_index,
                prepared_random=prepared_random,
            )
            return (
                sample,
                self._update_sampler_state(
                    block=block,
                    sample_key=sample_key,
                    sampler_state=sampler_state,
                    sample=sample,
                ),
            )

        return self._profile_call(
            "sample_single_block",
            lambda: self._profile_call(
                f"sample_single_block.block{block_index}",
                _sample,
            ),
        )

    def sample_single_block(self, key, block_index: int):
        sample_key, _ = jax.random.split(key, 2)
        sample, sampler_state = self._sample_single_block_from_sample_key(
            sample_key,
            block_index,
            sampler_state=_state_runtime.sampler_state_for_block(self, block_index),
        )
        _state_runtime.store_sampler_state_for_block(self, block_index, sampler_state)
        return sample

    def _after_sampling_group(
        self, group_index: int, sampling_group: Sequence[int]
    ) -> None:
        del group_index, sampling_group

    def run_sweep(
        self,
        key,
        *,
        sample_keys: Sequence[object] | None = None,
        prepared_randoms_by_block: dict[int, object] | None = None,
    ) -> None:
        def _run():
            self._require_state()
            resolved_prepared_randoms = prepared_randoms_by_block
            block_sample_keys = (
                tuple(sample_keys)
                if sample_keys is not None
                else tuple(
                    _sample_keys_for_iteration_keys(
                        (key,),
                        n_blocks=len(self.program.gibbs_spec.free_blocks),
                    )[0]
                )
            )
            if resolved_prepared_randoms is None:
                resolved_prepared_randoms = {}
                for block in self.compiled.blocks:
                    handler = self._parameter_family_handler(block)
                    prepared_random = handler.prepare_batch_sample_inputs(
                        self,
                        block,
                        (block_sample_keys[block.block_index],),
                    )
                    resolved_prepared_randoms[block.block_index] = handler.select_prepared_random(
                        self,
                        block,
                        prepared_random,
                        0,
                    )
            for group_index, sampling_group in enumerate(
                self.program.gibbs_spec.sampling_order
            ):
                pending_updates = {}
                pending_sampler_states = {}
                for block_index in sampling_group:
                    new_state, new_sampler_state = self._profile_call(
                        f"run_sweep.group{group_index}.sample.block{block_index}",
                        lambda block_index=block_index: self._sample_single_block_from_sample_key(
                            block_sample_keys[block_index],
                            block_index,
                            sampler_state=_state_runtime.sampler_state_for_block(
                                self, block_index
                            ),
                            prepared_random=(
                                resolved_prepared_randoms.get(block_index)
                            ),
                        ),
                    )
                    pending_updates[block_index] = new_state
                    pending_sampler_states[block_index] = new_sampler_state
                for block_index, new_state in pending_updates.items():
                    self._profile_call(
                        f"run_sweep.group{group_index}.write.block{block_index}",
                        lambda block_index=block_index, new_state=new_state: self._write_block_state(
                            block_index, new_state
                        ),
                    )
                    _state_runtime.store_sampler_state_for_block(
                        self,
                        block_index,
                        pending_sampler_states[block_index],
                    )
                self._after_sampling_group(group_index, sampling_group)

        self._profile_call("run_sweep", _run)

    def run_blocks(self, key, *, n_iters: int) -> None:
        if n_iters == 0:
            return
        iteration_keys = tuple(jax.random.split(key, n_iters))
        prepared_randoms = self._prepare_run_randoms(iteration_keys)
        for iter_index, iter_key in enumerate(iteration_keys):
            self.run_sweep(
                iter_key,
                sample_keys=prepared_randoms.sample_keys[iter_index],
                prepared_randoms_by_block={
                    block_index: self._parameter_family_handler(
                        self.compiled.blocks[block_index]
                    ).select_prepared_random(
                        self,
                        self.compiled.blocks[block_index],
                        prepared_random,
                        iter_index,
                    )
                    for block_index, prepared_random in prepared_randoms.prepared_randoms_by_block.items()
                },
            )

    def _sample_block_batch(self, sample_keys: Sequence[object], block_index: int):
        block = self.compiled.blocks[block_index]
        handler = self._parameter_family_handler(block)
        prepared_random = self._profile_call(
            f"sample_block_batch.block{block_index}.prepared_random",
            lambda: handler.prepare_batch_sample_inputs(self, block, sample_keys),
        )
        return self._sample_block(
            sample_keys[0],
            block_index,
            prepared_random=prepared_random,
        )

    def run_sweep_batch(
        self, iteration_keys: Sequence[object]
    ) -> None:
        if not iteration_keys:
            return

        def _run():
            self._require_state()
            if not all(
                block.sampler_lowering.sampler_state_spec is None
                and self._parameter_family_handler(block).supports_batch_sampling
                for block in self.compiled.blocks
            ):
                raise TypeError(
                    "run_sweep_batch() requires stateless blocks with explicit batched sampling support."
                )

            n_free_blocks = len(self.program.gibbs_spec.free_blocks)
            sample_keys = _sample_keys_for_iteration_keys(iteration_keys, n_blocks=n_free_blocks)
            block_sample_keys = tuple(
                tuple(sample_keys[iter_index][block_index] for iter_index in range(len(sample_keys)))
                for block_index in range(n_free_blocks)
            )
            for group_index, sampling_group in enumerate(
                self.program.gibbs_spec.sampling_order
            ):
                pending_updates = {
                    block_index: self._profile_call(
                        f"run_sweep_batch.group{group_index}.sample.block{block_index}",
                        lambda block_index=block_index: self._sample_block_batch(
                            block_sample_keys[block_index], block_index
                        ),
                    )
                    for block_index in sampling_group
                }
                for block_index, new_state in pending_updates.items():
                    self._profile_call(
                        f"run_sweep_batch.group{group_index}.write.block{block_index}",
                        lambda block_index=block_index, new_state=new_state: self._write_block_state(
                            block_index, new_state
                        ),
                    )

        self._profile_call("run_sweep_batch", _run)

    def run_blocks_batch(self, keys: Sequence[object], *, n_iters: int) -> None:
        if n_iters == 0 or not keys:
            return

        per_job_iteration_keys = tuple(
            tuple(jax.random.split(key, n_iters))
            for key in keys
        )
        for iter_index in range(n_iters):
            self.run_sweep_batch(
                [job_keys[iter_index] for job_keys in per_job_iteration_keys]
            )

    def run_prepared_randoms(
        self,
        prepared_randoms: PreparedRunRandoms,
        *,
        start: int = 0,
        count: int | None = None,
    ) -> None:
        sample_keys = prepared_randoms.sample_keys
        if count is None:
            stop = len(sample_keys)
        else:
            stop = start + count
        for iter_index in range(start, stop):
            iteration_sample_keys = prepared_randoms.sample_keys[iter_index]
            iter_key = prepared_randoms.iteration_keys[iter_index]
            self.run_sweep(
                iter_key,
                sample_keys=iteration_sample_keys,
                prepared_randoms_by_block={
                    block_index: self._parameter_family_handler(
                        self.compiled.blocks[block_index]
                    ).select_prepared_random(
                        self,
                        self.compiled.blocks[block_index],
                        prepared_random,
                        iter_index,
                    )
                    for block_index, prepared_random in prepared_randoms.prepared_randoms_by_block.items()
                },
            )

    def _run_prepared_schedule_warmup(
        self, prepared_randoms: PreparedScheduleRandoms
    ) -> None:
        if prepared_randoms.warmup_count <= 0:
            return
        self.run_prepared_randoms(
            prepared_randoms.run_randoms,
            start=0,
            count=prepared_randoms.warmup_count,
        )

    def _iter_prepared_schedule_sample_intervals(
        self, prepared_randoms: PreparedScheduleRandoms
    ):
        for offset in prepared_randoms.sample_interval_offsets:
            self.run_prepared_randoms(
                prepared_randoms.run_randoms,
                start=offset,
                count=prepared_randoms.steps_per_sample,
            )
            yield offset

    def sample_states(
        self,
        key,
        schedule: SamplingSchedule,
        *,
        nodes_to_sample: Sequence[Block],
        init_state_free,
        state_clamp,
    ):
        return _observation_runtime.sample_states(
            self,
            key,
            schedule,
            nodes_to_sample=nodes_to_sample,
            init_state_free=init_state_free,
            state_clamp=state_clamp,
        )

    def sample_observed_blocks(
        self,
        key,
        schedule: SamplingSchedule,
        *,
        blocks: Sequence[Block],
    ):
        return _observation_runtime.sample_observed_blocks(
            self,
            key,
            schedule,
            blocks=blocks,
        )

    def sample_loaded_observation(
        self,
        key,
        schedule: SamplingSchedule,
        *,
        state_clamp,
        observation_carry_init,
        f_observe: AbstractObserver,
    ):
        return _observation_runtime.sample_loaded_observation(
            self,
            key,
            schedule,
            state_clamp=state_clamp,
            observation_carry_init=observation_carry_init,
            f_observe=f_observe,
        )

    def sample_loaded_observation_jobs(
        self,
        schedule: SamplingSchedule,
        *,
        jobs: Sequence[LoadedObservationJob],
        f_observe: AbstractObserver,
    ):
        return _observation_runtime.sample_loaded_observation_jobs(
            self,
            schedule,
            jobs=jobs,
            f_observe=f_observe,
        )

    def sample_loaded_state_jobs(
        self,
        schedule: SamplingSchedule,
        *,
        jobs: Sequence[LoadedStateJob],
        nodes_to_sample: Sequence[Block] | None = None,
        observation_plan: CompiledObservationPlan | None = None,
    ):
        return _observation_runtime.sample_loaded_state_jobs(
            self,
            schedule,
            jobs=jobs,
            nodes_to_sample=nodes_to_sample,
            observation_plan=observation_plan,
        )

    def _run_loaded_jobs(self, jobs, *, run_loaded_job):
        return _observation_runtime._run_loaded_jobs(
            self,
            jobs,
            run_loaded_job=run_loaded_job,
        )

    def sample_moment_observer(
        self,
        key,
        schedule: SamplingSchedule,
        *,
        observer: MomentAccumulatorObserver,
        observation_carry_init,
        plan: CompiledMomentObserverPlan | None = None,
    ):
        return _observation_runtime.sample_moment_observer(
            self,
            key,
            schedule,
            observer=observer,
            observation_carry_init=observation_carry_init,
            plan=plan,
        )

    def sample_moment_observer_numpy(
        self,
        key,
        schedule: SamplingSchedule,
        *,
        observer: MomentAccumulatorObserver,
        observation_carry_init,
        plan: CompiledMomentObserverPlan | None = None,
    ):
        return _observation_runtime.sample_moment_observer_numpy(
            self,
            key,
            schedule,
            observer=observer,
            observation_carry_init=observation_carry_init,
            plan=plan,
        )

    def _sample_moment_observer_impl(
        self,
        key,
        schedule: SamplingSchedule,
        *,
        observer: MomentAccumulatorObserver,
        observation_carry_init,
        plan: CompiledMomentObserverPlan | None,
        to_jax: bool,
    ):
        return _observation_runtime._sample_moment_observer_impl(
            self,
            key,
            schedule,
            observer=observer,
            observation_carry_init=observation_carry_init,
            plan=plan,
            to_jax=to_jax,
        )

    def _supports_batched_spin_moment_jobs(self) -> bool:
        return _observation_runtime._supports_batched_spin_moment_jobs(self)

    def _accumulate_moment_observation_batch(
        self,
        observer: MomentAccumulatorObserver,
        plan: CompiledMomentObserverPlan,
        mem_batch,
        observed_state_batch,
    ):
        return _observation_runtime._accumulate_moment_observation_batch(
            self,
            observer,
            plan,
            mem_batch,
            observed_state_batch,
        )

    def _sample_loaded_batched_spin_moment_job_group_numpy(
        self,
        schedule: SamplingSchedule,
        *,
        jobs: Sequence[LoadedObservationJob],
        observer: MomentAccumulatorObserver,
        plan: CompiledMomentObserverPlan,
    ):
        return _observation_runtime._sample_loaded_batched_spin_moment_job_group_numpy(
            self,
            schedule,
            jobs=jobs,
            observer=observer,
            plan=plan,
        )

    def sample_loaded_numpy_moment_observation_jobs(
        self,
        schedule: SamplingSchedule,
        *,
        jobs: Sequence[LoadedObservationJob],
        observer: MomentAccumulatorObserver,
    ):
        return _observation_runtime.sample_loaded_numpy_moment_observation_jobs(
            self,
            schedule,
            jobs=jobs,
            observer=observer,
        )

    def sample_loaded_states(
        self,
        key,
        schedule: SamplingSchedule,
        *,
        nodes_to_sample: Sequence[Block] | None = None,
        observation_plan: CompiledObservationPlan | None = None,
    ):
        return _observation_runtime.sample_loaded_states(
            self,
            key,
            schedule,
            nodes_to_sample=nodes_to_sample,
            observation_plan=observation_plan,
        )
