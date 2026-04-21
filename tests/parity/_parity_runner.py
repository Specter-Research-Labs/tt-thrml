"""parity runner: upstream THRML (JAX reference) vs tt_thrml (TT hardware).

Run directly: python -m tests.parity._parity_runner
Requires: SYSTEM_DESC_PATH, TTMLIR_BUILD_DIR env vars, TT hardware device.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

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

# GaussianConditional is a tt-local extension not yet in upstream THRML.
# It is used for the upstream JAX reference side of gaussian/mixed scenarios.
from tt_thrml.conditional_samplers import GaussianConditional

import tt_thrml
from tt_thrml import TTMLIRConfig


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


def _clone_state_blocks(blocks: Sequence[object]) -> list[object]:
    return [jnp.asarray(np.asarray(block)).copy() for block in blocks]


def _sample_states_many(
    case: SampleCase,
    *,
    sample_keys,
    executor,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
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
        tt_outputs = executor.sample_states(
            sample_key,
            case.schedule,
            case.nodes_to_sample,
            init_state_free=_clone_state_blocks(case.init_state_free),
            state_clamp=_clone_state_blocks(case.state_clamp),
        )
        upstream_samples.append([np.asarray(o) for o in upstream_outputs])
        tt_samples.append([np.asarray(o) for o in tt_outputs])
    return upstream_samples, tt_samples


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


def run_spin_scenario(ttnn, device, config: TTMLIRConfig) -> dict:
    case = _make_spin_case()
    executor = tt_thrml.make_executor(ttnn, device, case.program, config)
    sample_keys = jax.random.split(jax.random.key(5101), 6)
    upstream_many, tt_many = _sample_states_many(
        case, sample_keys=sample_keys, executor=executor
    )
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


def run_categorical_scenario(ttnn, device, config: TTMLIRConfig) -> dict:
    n_categories = 3
    case = _make_categorical_case(n_categories=n_categories)
    executor = tt_thrml.make_executor(ttnn, device, case.program, config)
    sample_keys = jax.random.split(jax.random.key(5202), 12)
    upstream_many, tt_many = _sample_states_many(
        case, sample_keys=sample_keys, executor=executor
    )
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
            _empirical_probs(upstream_samples[:, i], n_states=n_categories)
            for i in range(upstream_samples.shape[1])
        ],
        axis=0,
    )
    tt_node_hist = np.stack(
        [
            _empirical_probs(tt_samples[:, i], n_states=n_categories)
            for i in range(tt_samples.shape[1])
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


def run_gaussian_scenario(ttnn, device, config: TTMLIRConfig) -> dict:
    case = _make_gaussian_case()
    executor = tt_thrml.make_executor(ttnn, device, case.program, config)
    sample_keys = jax.random.split(jax.random.key(5303), 6)
    upstream_many, tt_many = _sample_states_many(
        case, sample_keys=sample_keys, executor=executor
    )
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


def run_mixed_scenario(ttnn, device, config: TTMLIRConfig) -> dict:
    n_categories = 3
    case = _make_mixed_case(n_categories=n_categories)
    executor = tt_thrml.make_executor(ttnn, device, case.program, config)
    sample_keys = jax.random.split(jax.random.key(5404), 6)
    upstream_many, tt_many = _sample_states_many(
        case, sample_keys=sample_keys, executor=executor
    )
    upstream_spin = _stack_output(upstream_many, 0)
    tt_spin = _stack_output(tt_many, 0)
    upstream_cat = _stack_output(upstream_many, 1)
    tt_cat = _stack_output(tt_many, 1)
    upstream_gaussian = _stack_output(upstream_many, 2).astype(np.float64)
    tt_gaussian = _stack_output(tt_many, 2).astype(np.float64)
    upstream_discrete_hist = _empirical_probs(
        _joint_spin_categorical_code(upstream_spin, upstream_cat, n_categories=n_categories),
        n_states=(2 ** upstream_spin.shape[1]) * (n_categories ** upstream_cat.shape[1]),
    )
    tt_discrete_hist = _empirical_probs(
        _joint_spin_categorical_code(tt_spin, tt_cat, n_categories=n_categories),
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


def run_observation_clamp_scenario(ttnn, device, config: TTMLIRConfig) -> dict:
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

    executor = tt_thrml.make_executor(ttnn, device, program, config)
    sample_keys = jax.random.split(jax.random.key(5505), 4)

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
        tt_run, _ = executor.sample_with_observation(
            sample_key,
            schedule,
            observer,
            init_state_free=_clone_state_blocks(init_state_free),
            state_clamp=_clone_state_blocks(state_clamp),
        )
        if upstream_carry is None:
            upstream_carry = [np.asarray(e).copy() for e in upstream_run]
            tt_carry = [np.asarray(e).copy() for e in tt_run]
        else:
            for i in range(len(upstream_carry)):
                upstream_carry[i] += np.asarray(upstream_run[i])
                tt_carry[i] += np.asarray(tt_run[i])

    total_samples = schedule.n_samples * len(sample_keys)
    upstream_moments = [e / total_samples for e in upstream_carry]
    tt_moments = [e / total_samples for e in tt_carry]
    return {
        "sample_count": int(total_samples),
        "diffs": {
            "first_moment_linf": _metric_linf(upstream_moments[0], tt_moments[0]),
            "second_moment_linf": _metric_linf(upstream_moments[1], tt_moments[1]),
        },
    }


def _require_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for parity runs.")
    return Path(value).resolve()


def _artifact_root() -> Path:
    root = os.environ.get("TT_THRML_PARITY_ARTIFACT_ROOT")
    if root:
        return Path(root).resolve()
    return Path("/tmp/tt-thrml-artifacts/wormhole-parity").resolve()


def main() -> None:
    import ttnn

    system_desc_path = _require_env_path("SYSTEM_DESC_PATH")
    build_dir = _require_env_path("TTMLIR_BUILD_DIR")
    artifact_root = _artifact_root()
    artifact_root.mkdir(parents=True, exist_ok=True)

    device_id_raw = os.environ.get("TT_THRML_TEST_DEVICE_IDS", "0")
    device_id = int(device_id_raw.split(",")[0].strip() or "0")

    config = tt_thrml.make_ttmlir_config(
        system_desc_path=system_desc_path,
        artifact_root=artifact_root,
        build_dir=build_dir,
    )

    device = tt_thrml.open_device(ttnn, device_id=device_id)
    try:
        results = {
            "spin": run_spin_scenario(ttnn, device, config),
            "categorical": run_categorical_scenario(ttnn, device, config),
            "gaussian": run_gaussian_scenario(ttnn, device, config),
            "mixed": run_mixed_scenario(ttnn, device, config),
            "observation_clamp": run_observation_clamp_scenario(ttnn, device, config),
        }
        print(json.dumps(results, sort_keys=True))
    finally:
        tt_thrml.close_device(ttnn, device)


if __name__ == "__main__":
    main()
