from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:
    import torch  # type: ignore
except ImportError:
    from ._torch_stub import install_torch_stub

    torch = install_torch_stub()

import equinox as eqx
import jax
import jax.numpy as jnp
from thrml.block_management import Block
from thrml.block_sampling import (
    BlockGibbsSpec,
    SamplingSchedule,
    sample_states as upstream_sample_states,
    sample_with_observation as upstream_sample_with_observation,
)
from thrml.factor import AbstractFactor, FactorSamplingProgram
from thrml.interaction import InteractionGroup
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    DiscreteEBMFactor,
    SpinEBMFactor,
    SpinGibbsConditional,
)
from thrml.observers import MomentAccumulatorObserver
from thrml.pgm import AbstractNode, CategoricalNode, SpinNode

import tt_thrml
import tt_thrml.runtime.backend_executor as backend_executor
from tt_thrml.compiler.categorical_ops import reference_categorical_theta_op
from tt_thrml.compiler.gaussian_ops import reference_gaussian_canonical_op
from tt_thrml.compiler.spin_ops import reference_spin_gamma_op
from tt_thrml.conditional_samplers import GaussianConditional
from tt_thrml.runtime_config import (
    CATEGORICAL_PARAMETER_FAMILY,
    GAUSSIAN_PARAMETER_FAMILY,
    SPIN_PARAMETER_FAMILY,
    ParameterKernelBackend,
    make_backend_binding,
)


def _install_parity_program_cache_identity():
    # Golden parity only needs semantic execution, not production cache fingerprints.
    backend_executor._program_cache_identity = (  # type: ignore[assignment]
        lambda program: (id(program), f"parity:{id(program)}")
    )


_install_parity_program_cache_identity()


class NumpyTTNN:
    Tensor = torch.Tensor
    bfloat16 = torch.float32
    uint32 = getattr(torch, "uint32", torch.int64)
    int32 = getattr(torch, "int32", torch.int64)
    ROW_MAJOR_LAYOUT = "row_major"
    TILE_LAYOUT = "tile"

    def from_torch(self, value, *, dtype=None, layout=None, device=None, mesh_mapper=None):
        del layout, device, mesh_mapper
        tensor = value.clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return tensor.to(dtype) if dtype is not None else tensor

    def to_torch(self, value):
        return value.clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value)

    def full(self, shape, *, fill_value, dtype=None, layout=None, device=None):
        del layout, device
        return torch.full(shape, fill_value=fill_value, dtype=dtype or torch.float32)

    def repeat(self, value, sizes):
        return value.repeat(sizes)

    def concat(self, values, dim=0):
        return torch.concat(values, dim=dim)

    def gather(self, values, dim, *, index):
        return torch.gather(values, dim, index.to(torch.int64))

    def multiply(self, lhs, rhs):
        return lhs * rhs

    def sum(self, value, *, dim, keepdim):
        return torch.sum(value, dim=dim, keepdim=keepdim)

    def add(self, lhs, rhs):
        return lhs + rhs

    def reciprocal(self, value):
        return torch.reciprocal(value)

    def sqrt(self, value):
        return torch.sqrt(value)

    def argmax(self, value, dim=None, keepdim=False, **kwargs):
        del kwargs
        return torch.argmax(value, dim=dim, keepdim=keepdim)

    def gt(self, lhs, rhs):
        return lhs > rhs

    def where(self, condition, lhs, rhs):
        return torch.where(condition, lhs, rhs)

    def reshape(self, value, shape):
        return value.reshape(shape)

    def to_layout(self, value, layout):
        del layout
        return value

    def typecast(self, value, *, dtype):
        return value.to(dtype)

    def to_dtype(self, value, dtype):
        return value.to(dtype)


class ContinuousNode(AbstractNode):
    pass


class LinearInteraction(eqx.Module):
    weights: jax.Array


class QuadraticInteraction(eqx.Module):
    inverse_weights: jax.Array


class LinearFactor(AbstractFactor):
    weights: jax.Array

    def __init__(self, weights: jax.Array, block: Block):
        super().__init__([block])
        self.weights = weights

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                interaction=LinearInteraction(self.weights),
                head_nodes=self.node_groups[0],
                tail_nodes=[],
            )
        ]


class QuadraticFactor(AbstractFactor):
    inverse_weights: jax.Array

    def __init__(self, inverse_weights: jax.Array, block: Block):
        super().__init__([block])
        self.inverse_weights = inverse_weights

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                interaction=QuadraticInteraction(self.inverse_weights),
                head_nodes=self.node_groups[0],
                tail_nodes=[],
            )
        ]


class CouplingFactor(AbstractFactor):
    weights: jax.Array

    def __init__(self, weights: jax.Array, blocks: tuple[Block, Block]):
        super().__init__(list(blocks))
        self.weights = weights

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                interaction=LinearInteraction(self.weights),
                head_nodes=self.node_groups[0],
                tail_nodes=[self.node_groups[1]],
            ),
            InteractionGroup(
                interaction=LinearInteraction(self.weights),
                head_nodes=self.node_groups[1],
                tail_nodes=[self.node_groups[0]],
            ),
        ]


@dataclass(frozen=True)
class SampleCase:
    program: FactorSamplingProgram
    schedule: SamplingSchedule
    init_state_free: list[object]
    state_clamp: list[object]
    nodes_to_sample: list[Block]


def _reference_backend():
    ttnn = NumpyTTNN()
    return make_backend_binding(
        ttnn,
        "fake:0",
        parameter_kernel_ops={
            SPIN_PARAMETER_FAMILY: reference_spin_gamma_op,
            CATEGORICAL_PARAMETER_FAMILY: reference_categorical_theta_op,
            GAUSSIAN_PARAMETER_FAMILY: reference_gaussian_canonical_op,
        },
        parameter_kernel_backends={
            SPIN_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM,
            CATEGORICAL_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM,
            GAUSSIAN_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM,
        },
    )


def _clone_state_blocks(blocks: Sequence[object]) -> list[object]:
    return [jnp.asarray(np.asarray(block)).copy() for block in blocks]


def _sample_states_many(case: SampleCase, *, sample_keys) -> tuple[list[np.ndarray], list[np.ndarray]]:
    backend = _reference_backend()
    tt_thrml.clear_compiled_program_cache()
    try:
        upstream_samples = []
        tt_samples = []
        for sample_key in sample_keys:
            upstream_outputs = upstream_sample_states(
                sample_key,
                case.program,
                case.schedule,
                _clone_state_blocks(case.init_state_free),
                _clone_state_blocks(case.state_clamp),
                case.nodes_to_sample,
            )
            tt_outputs = tt_thrml.sample_states(
                sample_key,
                case.program,
                case.schedule,
                _clone_state_blocks(case.init_state_free),
                _clone_state_blocks(case.state_clamp),
                case.nodes_to_sample,
                backend=backend,
            )
            upstream_samples.append(
                [np.asarray(output) for output in upstream_outputs]
            )
            tt_samples.append([np.asarray(output) for output in tt_outputs])
        return upstream_samples, tt_samples
    finally:
        tt_thrml.clear_compiled_program_cache()


def _spin_state_code(samples: np.ndarray) -> np.ndarray:
    weights = (1 << np.arange(samples.shape[1], dtype=np.int64)).reshape(1, -1)
    return np.sum(samples.astype(np.int64) * weights, axis=1)


def _categorical_state_code(samples: np.ndarray, *, base: int) -> np.ndarray:
    weights = (base ** np.arange(samples.shape[1], dtype=np.int64)).reshape(1, -1)
    return np.sum(samples.astype(np.int64) * weights, axis=1)


def _joint_spin_categorical_code(
    spin_samples: np.ndarray,
    categorical_samples: np.ndarray,
    *,
    n_categories: int,
) -> np.ndarray:
    categorical_code = _categorical_state_code(categorical_samples, base=n_categories)
    spin_code = _spin_state_code(spin_samples)
    return categorical_code * (2 ** spin_samples.shape[1]) + spin_code


def _empirical_probs(encoded_samples: np.ndarray, *, n_states: int) -> np.ndarray:
    counts = np.bincount(encoded_samples.astype(np.int64), minlength=n_states).astype(np.float64)
    return counts / counts.sum()


def _signed_spin(samples: np.ndarray) -> np.ndarray:
    return np.where(samples.astype(np.bool_), 1.0, -1.0)


def _gaussian_mean_and_cov(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = samples.mean(axis=0)
    centered = samples - mean
    cov = (centered.T @ centered) / float(samples.shape[0])
    return mean, cov


def _metric_linf(lhs, rhs) -> float:
    return float(np.max(np.abs(np.asarray(lhs) - np.asarray(rhs))))


def _stack_output(samples_many: list[list[np.ndarray]], output_index: int) -> np.ndarray:
    return np.concatenate(
        [sample_set[output_index] for sample_set in samples_many],
        axis=0,
    )


def _make_spin_case() -> SampleCase:
    nodes = [SpinNode() for _ in range(4)]
    bias_factor = SpinEBMFactor(
        [Block(nodes)],
        jnp.asarray([0.35, -0.25, 0.2, -0.15], dtype=jnp.float32),
    )
    pair_factor = SpinEBMFactor(
        [Block(nodes[:-1]), Block(nodes[1:])],
        jnp.asarray([0.75, -0.45, 0.3], dtype=jnp.float32),
    )
    free_blocks = [Block(nodes[0::2]), Block(nodes[1::2])]
    program = FactorSamplingProgram(
        BlockGibbsSpec(free_blocks, []),
        [SpinGibbsConditional(), SpinGibbsConditional()],
        [bias_factor, pair_factor],
        [],
    )
    init_key = jax.random.key(1001)
    init_state_free = [
        jax.random.bernoulli(block_key, 0.5, (len(block.nodes),)).astype(jnp.bool_)
        for block_key, block in zip(
            jax.random.split(init_key, len(free_blocks)),
            free_blocks,
            strict=True,
        )
    ]
    return SampleCase(
        program=program,
        schedule=SamplingSchedule(n_warmup=12, n_samples=96, steps_per_sample=2),
        init_state_free=init_state_free,
        state_clamp=[],
        nodes_to_sample=[Block(nodes)],
    )


def _make_categorical_case(*, n_categories: int = 3) -> SampleCase:
    nodes = [CategoricalNode() for _ in range(4)]
    bias_factor = CategoricalEBMFactor(
        [Block(nodes)],
        jnp.asarray(
            [
                [0.4, -0.15, 0.0],
                [-0.1, 0.25, -0.2],
                [0.05, -0.35, 0.3],
                [0.2, -0.05, -0.1],
            ],
            dtype=jnp.float32,
        ),
    )
    pair_factor = CategoricalEBMFactor(
        [Block(nodes[:-1]), Block(nodes[1:])],
        jnp.asarray(
            [
                [[0.9, -0.3, 0.2], [-0.1, 0.4, -0.2], [0.15, -0.05, 0.25]],
                [[0.25, -0.4, 0.1], [0.35, 0.2, -0.3], [-0.15, 0.1, 0.45]],
                [[0.5, -0.2, -0.1], [0.05, 0.3, -0.25], [-0.2, 0.15, 0.35]],
            ],
            dtype=jnp.float32,
        ),
    )
    free_blocks = [Block(nodes[0::2]), Block(nodes[1::2])]
    sampler = CategoricalGibbsConditional(n_categories)
    program = FactorSamplingProgram(
        BlockGibbsSpec(free_blocks, []),
        [sampler, sampler],
        [bias_factor, pair_factor],
        [],
    )
    init_key = jax.random.key(2002)
    init_state_free = [
        jax.random.randint(
            block_key,
            shape=(len(block.nodes),),
            minval=0,
            maxval=n_categories,
            dtype=jnp.uint8,
        )
        for block_key, block in zip(
            jax.random.split(init_key, len(free_blocks)),
            free_blocks,
            strict=True,
        )
    ]
    return SampleCase(
        program=program,
        schedule=SamplingSchedule(n_warmup=12, n_samples=256, steps_per_sample=2),
        init_state_free=init_state_free,
        state_clamp=[],
        nodes_to_sample=[Block(nodes)],
    )


def _make_gaussian_case() -> SampleCase:
    nodes = [ContinuousNode() for _ in range(3)]
    free_blocks = [Block(nodes[0::2]), Block(nodes[1::2])]
    node_shape_dtypes = {
        ContinuousNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
    }
    program = FactorSamplingProgram(
        BlockGibbsSpec(free_blocks, [], node_shape_dtypes),
        [GaussianConditional(), GaussianConditional()],
        [
            LinearFactor(
                jnp.asarray([0.15, -0.35, 0.2], dtype=jnp.float32),
                Block(nodes),
            ),
            QuadraticFactor(
                jnp.asarray([0.7, 0.9, 0.8], dtype=jnp.float32),
                Block(nodes),
            ),
            CouplingFactor(
                jnp.asarray([0.12, -0.08], dtype=jnp.float32),
                (Block(nodes[:-1]), Block(nodes[1:])),
            ),
        ],
        [],
    )
    init_key = jax.random.key(3003)
    init_state_free = [
        0.2 * jax.random.normal(block_key, (len(block.nodes),), dtype=jnp.float32)
        for block_key, block in zip(
            jax.random.split(init_key, len(free_blocks)),
            free_blocks,
            strict=True,
        )
    ]
    return SampleCase(
        program=program,
        schedule=SamplingSchedule(n_warmup=16, n_samples=128, steps_per_sample=3),
        init_state_free=init_state_free,
        state_clamp=[],
        nodes_to_sample=[Block(nodes)],
    )


def _make_mixed_case(*, n_categories: int = 3) -> SampleCase:
    spin_nodes = [SpinNode() for _ in range(2)]
    categorical_nodes = [CategoricalNode() for _ in range(2)]
    continuous_nodes = [ContinuousNode() for _ in range(2)]
    free_super_blocks = [
        (
            Block([spin_nodes[0]]),
            Block([categorical_nodes[0]]),
            Block([continuous_nodes[0]]),
        ),
        (
            Block([spin_nodes[1]]),
            Block([categorical_nodes[1]]),
            Block([continuous_nodes[1]]),
        ),
    ]
    node_shape_dtypes = {
        SpinNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool_),
        CategoricalNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.uint8),
        ContinuousNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    }
    program = FactorSamplingProgram(
        BlockGibbsSpec(free_super_blocks, [], node_shape_dtypes),
        [
            SpinGibbsConditional(),
            CategoricalGibbsConditional(n_categories),
            GaussianConditional(),
            SpinGibbsConditional(),
            CategoricalGibbsConditional(n_categories),
            GaussianConditional(),
        ],
        [
            SpinEBMFactor(
                [Block(spin_nodes)],
                jnp.asarray([0.25, -0.2], dtype=jnp.float32),
            ),
            CategoricalEBMFactor(
                [Block(categorical_nodes)],
                jnp.asarray(
                    [[0.2, -0.1, 0.0], [-0.15, 0.3, -0.05]],
                    dtype=jnp.float32,
                ),
            ),
            DiscreteEBMFactor(
                [Block(spin_nodes)],
                [Block(categorical_nodes)],
                jnp.asarray(
                    [[0.55, -0.25, 0.15], [-0.2, 0.4, 0.1]],
                    dtype=jnp.float32,
                ),
            ),
            LinearFactor(
                jnp.asarray([0.1, -0.2], dtype=jnp.float32),
                Block(continuous_nodes),
            ),
            QuadraticFactor(
                jnp.asarray([0.85, 0.75], dtype=jnp.float32),
                Block(continuous_nodes),
            ),
            CouplingFactor(
                jnp.asarray([0.11], dtype=jnp.float32),
                (Block([continuous_nodes[0]]), Block([continuous_nodes[1]])),
            ),
        ],
        [],
    )
    init_key = jax.random.key(4004)
    init_keys = jax.random.split(init_key, 6)
    init_state_free = [
        jax.random.bernoulli(init_keys[0], 0.5, (1,)).astype(jnp.bool_),
        jax.random.randint(
            init_keys[1],
            shape=(1,),
            minval=0,
            maxval=n_categories,
            dtype=jnp.uint8,
        ),
        0.2 * jax.random.normal(init_keys[2], (1,), dtype=jnp.float32),
        jax.random.bernoulli(init_keys[3], 0.5, (1,)).astype(jnp.bool_),
        jax.random.randint(
            init_keys[4],
            shape=(1,),
            minval=0,
            maxval=n_categories,
            dtype=jnp.uint8,
        ),
        0.2 * jax.random.normal(init_keys[5], (1,), dtype=jnp.float32),
    ]
    return SampleCase(
        program=program,
        schedule=SamplingSchedule(n_warmup=14, n_samples=96, steps_per_sample=2),
        init_state_free=init_state_free,
        state_clamp=[],
        nodes_to_sample=[
            Block(spin_nodes),
            Block(categorical_nodes),
            Block(continuous_nodes),
        ],
    )


def _spin_observer(state, _blocks):
    return [2 * x.astype(jnp.int8) - 1 for x in state]


def _run_spin_scenario():
    case = _make_spin_case()
    sample_keys = jax.random.split(jax.random.key(5101), 6)
    upstream_many, tt_many = _sample_states_many(case, sample_keys=sample_keys)
    upstream_samples = _stack_output(upstream_many, 0)
    tt_samples = _stack_output(tt_many, 0)
    upstream_hist = _empirical_probs(_spin_state_code(upstream_samples), n_states=16)
    tt_hist = _empirical_probs(_spin_state_code(tt_samples), n_states=16)
    upstream_mean = _signed_spin(upstream_samples).mean(axis=0)
    tt_mean = _signed_spin(tt_samples).mean(axis=0)
    return {
        "sample_count": int(upstream_samples.shape[0]),
        "diffs": {
            "state_hist_linf": _metric_linf(upstream_hist, tt_hist),
            "mean_signed_linf": _metric_linf(upstream_mean, tt_mean),
        },
    }


def _run_categorical_scenario():
    n_categories = 3
    case = _make_categorical_case(n_categories=n_categories)
    sample_keys = jax.random.split(jax.random.key(5202), 12)
    upstream_many, tt_many = _sample_states_many(case, sample_keys=sample_keys)
    upstream_samples = _stack_output(upstream_many, 0)
    tt_samples = _stack_output(tt_many, 0)
    upstream_hist = _empirical_probs(
        _categorical_state_code(upstream_samples, base=n_categories),
        n_states=n_categories ** upstream_samples.shape[1],
    )
    tt_hist = _empirical_probs(
        _categorical_state_code(tt_samples, base=n_categories),
        n_states=n_categories ** tt_samples.shape[1],
    )
    upstream_node_hist = np.stack(
        [
            _empirical_probs(upstream_samples[:, node_index], n_states=n_categories)
            for node_index in range(upstream_samples.shape[1])
        ],
        axis=0,
    )
    tt_node_hist = np.stack(
        [
            _empirical_probs(tt_samples[:, node_index], n_states=n_categories)
            for node_index in range(tt_samples.shape[1])
        ],
        axis=0,
    )
    return {
        "sample_count": int(upstream_samples.shape[0]),
        "diffs": {
            "state_hist_linf": _metric_linf(upstream_hist, tt_hist),
            "node_hist_linf": _metric_linf(upstream_node_hist, tt_node_hist),
        },
    }


def _run_gaussian_scenario():
    case = _make_gaussian_case()
    sample_keys = jax.random.split(jax.random.key(5303), 6)
    upstream_many, tt_many = _sample_states_many(case, sample_keys=sample_keys)
    upstream_samples = _stack_output(upstream_many, 0).astype(np.float64)
    tt_samples = _stack_output(tt_many, 0).astype(np.float64)
    upstream_mean, upstream_cov = _gaussian_mean_and_cov(upstream_samples)
    tt_mean, tt_cov = _gaussian_mean_and_cov(tt_samples)
    return {
        "sample_count": int(upstream_samples.shape[0]),
        "diffs": {
            "mean_linf": _metric_linf(upstream_mean, tt_mean),
            "cov_linf": _metric_linf(upstream_cov, tt_cov),
        },
    }


def _run_mixed_scenario():
    n_categories = 3
    case = _make_mixed_case(n_categories=n_categories)
    sample_keys = jax.random.split(jax.random.key(5404), 6)
    upstream_many, tt_many = _sample_states_many(case, sample_keys=sample_keys)
    upstream_spin = _stack_output(upstream_many, 0)
    tt_spin = _stack_output(tt_many, 0)
    upstream_cat = _stack_output(upstream_many, 1)
    tt_cat = _stack_output(tt_many, 1)
    upstream_gaussian = _stack_output(upstream_many, 2).astype(np.float64)
    tt_gaussian = _stack_output(tt_many, 2).astype(np.float64)
    upstream_discrete_hist = _empirical_probs(
        _joint_spin_categorical_code(
            upstream_spin,
            upstream_cat,
            n_categories=n_categories,
        ),
        n_states=(2 ** upstream_spin.shape[1]) * (n_categories ** upstream_cat.shape[1]),
    )
    tt_discrete_hist = _empirical_probs(
        _joint_spin_categorical_code(
            tt_spin,
            tt_cat,
            n_categories=n_categories,
        ),
        n_states=(2 ** tt_spin.shape[1]) * (n_categories ** tt_cat.shape[1]),
    )
    upstream_mean, upstream_cov = _gaussian_mean_and_cov(upstream_gaussian)
    tt_mean, tt_cov = _gaussian_mean_and_cov(tt_gaussian)
    return {
        "sample_count": int(upstream_spin.shape[0]),
        "diffs": {
            "discrete_hist_linf": _metric_linf(upstream_discrete_hist, tt_discrete_hist),
            "gaussian_mean_linf": _metric_linf(upstream_mean, tt_mean),
            "gaussian_cov_linf": _metric_linf(upstream_cov, tt_cov),
        },
    }


def _run_observation_clamp_scenario():
    nodes = [SpinNode() for _ in range(3)]
    free_blocks = [Block([nodes[0]]), Block([nodes[1]])]
    clamped_blocks = [Block([nodes[2]])]
    program = FactorSamplingProgram(
        BlockGibbsSpec(free_blocks, clamped_blocks),
        [SpinGibbsConditional(), SpinGibbsConditional()],
        [
            SpinEBMFactor(
                [Block(nodes)],
                jnp.asarray([0.15, -0.25, 0.4], dtype=jnp.float32),
            ),
            SpinEBMFactor(
                [Block(nodes[:-1]), Block(nodes[1:])],
                jnp.asarray([0.55, -0.35], dtype=jnp.float32),
            ),
        ],
        [],
    )
    schedule = SamplingSchedule(n_warmup=10, n_samples=96, steps_per_sample=2)
    init_state_free = [
        jnp.asarray([True], dtype=jnp.bool_),
        jnp.asarray([False], dtype=jnp.bool_),
    ]
    state_clamp = [jnp.asarray([True], dtype=jnp.bool_)]
    observer = MomentAccumulatorObserver(
        (
            [(node,) for node in nodes],
            [(nodes[0], nodes[1]), (nodes[1], nodes[2])],
        ),
        _spin_observer,
    )
    sample_keys = jax.random.split(jax.random.key(5505), 4)
    backend = _reference_backend()
    tt_thrml.clear_compiled_program_cache()
    try:
        upstream_carry = None
        tt_carry = None
        for sample_key in sample_keys:
            upstream_run = upstream_sample_with_observation(
                sample_key,
                program,
                schedule,
                _clone_state_blocks(init_state_free),
                _clone_state_blocks(state_clamp),
                observer.init(),
                observer,
            )[0]
            tt_run = tt_thrml.sample_with_observation(
                sample_key,
                program,
                schedule,
                _clone_state_blocks(init_state_free),
                _clone_state_blocks(state_clamp),
                observer.init(),
                observer,
                backend=backend,
            )[0]
            if upstream_carry is None:
                upstream_carry = [np.asarray(entry).copy() for entry in upstream_run]
                tt_carry = [np.asarray(entry).copy() for entry in tt_run]
            else:
                for carry_index in range(len(upstream_carry)):
                    upstream_carry[carry_index] += np.asarray(upstream_run[carry_index])
                    tt_carry[carry_index] += np.asarray(tt_run[carry_index])
    finally:
        tt_thrml.clear_compiled_program_cache()
    total_samples = schedule.n_samples * len(sample_keys)
    upstream_moments = [entry / total_samples for entry in upstream_carry]
    tt_moments = [entry / total_samples for entry in tt_carry]
    return {
        "sample_count": int(total_samples),
        "diffs": {
            "first_moment_linf": _metric_linf(upstream_moments[0], tt_moments[0]),
            "second_moment_linf": _metric_linf(upstream_moments[1], tt_moments[1]),
        },
    }


def main() -> None:
    results = {
        "spin": _run_spin_scenario(),
        "categorical": _run_categorical_scenario(),
        "gaussian": _run_gaussian_scenario(),
        "mixed": _run_mixed_scenario(),
        "observation_clamp": _run_observation_clamp_scenario(),
    }
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
