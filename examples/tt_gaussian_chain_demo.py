"""Run a small continuous Gaussian THRML program on TT hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule
from thrml.factor import AbstractFactor, FactorSamplingProgram
from thrml.interaction import InteractionGroup
from thrml.pgm import AbstractNode
import ttnn

import tt_thrml
from tt_thrml.conditional_samplers import GaussianConditional
from _tt_demo_common import default_artifact_root, make_backend

DEMO_NAME = "gaussian-chain"


class ContinuousNode(AbstractNode):
    pass


class LinearInteraction(eqx.Module):
    weights: jax.Array


class QuadraticInteraction(eqx.Module):
    inverse_weights: jax.Array


class LinearFactor(AbstractFactor):
    weights: jax.Array

    def __init__(self, weights, block: Block):
        super().__init__([block])
        self.weights = weights

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                LinearInteraction(self.weights),
                self.node_groups[0],
                [],
            )
        ]


class QuadraticFactor(AbstractFactor):
    inverse_weights: jax.Array

    def __init__(self, inverse_weights, block: Block):
        super().__init__([block])
        self.inverse_weights = inverse_weights

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                QuadraticInteraction(self.inverse_weights),
                self.node_groups[0],
                [],
            )
        ]


class CouplingFactor(AbstractFactor):
    weights: jax.Array

    def __init__(self, weights, blocks: tuple[Block, Block]):
        super().__init__(list(blocks))
        self.weights = weights

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                LinearInteraction(self.weights),
                self.node_groups[0],
                [self.node_groups[1]],
            ),
            InteractionGroup(
                LinearInteraction(self.weights),
                self.node_groups[1],
                [self.node_groups[0]],
            ),
        ]


def _build_program(*, key):
    nodes = [ContinuousNode() for _ in range(4)]
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    node_shape_dtypes = {ContinuousNode: jax.ShapeDtypeStruct((), jnp.float32)}
    gibbs_spec = BlockGibbsSpec(free_blocks, [], node_shape_dtypes)

    key, subkey = jax.random.split(key)
    linear_weights = jax.random.normal(subkey, (len(nodes), 1), dtype=jnp.float32) * 0.2
    inverse_weights = jnp.ones((len(nodes), 1), dtype=jnp.float32) * 0.75
    key, subkey = jax.random.split(key)
    coupling_weights = jax.random.normal(
        subkey,
        (len(nodes) - 1, 1),
        dtype=jnp.float32,
    ) * 0.1

    program = FactorSamplingProgram(
        gibbs_spec,
        [GaussianConditional(), GaussianConditional()],
        [
            LinearFactor(linear_weights, Block(nodes)),
            QuadraticFactor(inverse_weights, Block(nodes)),
            CouplingFactor(coupling_weights, (Block(nodes[:-1]), Block(nodes[1:]))),
        ],
        [],
    )

    init_keys = jax.random.split(key, len(free_blocks) + 1)
    init_state_free = [
        jax.random.normal(
            init_keys[0],
            (len(free_blocks[0].nodes),),
            dtype=jnp.float32,
        ),
        jax.random.normal(
            init_keys[1],
            (len(free_blocks[1].nodes),),
            dtype=jnp.float32,
        ),
    ]
    return nodes, program, init_state_free, []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--parameter-kernels",
        choices=("ttmlir",),
        default="ttmlir",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--n-warmup", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--steps-per-sample", type=int, default=2)
    parser.add_argument("--system-desc-path", type=Path, default=None)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=default_artifact_root(DEMO_NAME),
    )
    args = parser.parse_args()

    build_key = jax.random.key(args.seed)
    nodes, program, init_state_free, state_clamp = _build_program(key=build_key)
    schedule = SamplingSchedule(
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        steps_per_sample=args.steps_per_sample,
    )
    sample_root = jax.random.fold_in(build_key, 29)
    nodes_to_sample = [Block(nodes)]

    device = tt_thrml.open_device(ttnn, device_id=args.device_id)
    backend = make_backend(
        ttnn_module=ttnn,
        device=device,
        parameter_kernels=args.parameter_kernels,
        system_desc_path=args.system_desc_path,
        artifact_root=args.artifact_root,
    )
    try:
        runs = []
        for run_index in range(args.repeat):
            key = jax.random.fold_in(sample_root, run_index)
            start = time.perf_counter()
            outputs = tt_thrml.sample_states(
                key,
                program,
                schedule,
                init_state_free,
                state_clamp,
                nodes_to_sample,
                backend=backend,
            )
            elapsed = time.perf_counter() - start
            samples = np.asarray(outputs[0], dtype=np.float32)
            runs.append(
                {
                    "run_index": run_index,
                    "elapsed_seconds": elapsed,
                    "samples_shape": list(samples.shape),
                    "samples_per_second": float(samples.shape[0] / elapsed),
                    "per_node_mean": samples.mean(axis=0).tolist(),
                    "per_node_std": samples.std(axis=0).tolist(),
                }
            )

        print(
            json.dumps(
                {
                    "demo": "gaussian_chain",
                    "backend": args.parameter_kernels,
                    "device_id": args.device_id,
                    "schedule": {
                        "n_warmup": args.n_warmup,
                        "n_samples": args.n_samples,
                        "steps_per_sample": args.steps_per_sample,
                    },
                    "runs": runs,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        tt_thrml.close_devices(ttnn, (device,))


if __name__ == "__main__":
    main()
