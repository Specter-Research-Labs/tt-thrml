"""Run a small categorical discrete-EBM THRML program on Tenstorrent hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
from jax import numpy as jnp
import numpy as np
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    SquareCategoricalEBMFactor,
)
from thrml.models.ebm import FactorizedEBM
from thrml.pgm import CategoricalNode
import ttnn

import tt_thrml
from _tt_demo_common import default_artifact_root, make_backend

DEMO_NAME = "categorical-ebm"


def _build_program(*, key, n_categories: int):
    nodes = [CategoricalNode() for _ in range(5)]
    extra_node = CategoricalNode()

    key, subkey = jax.random.split(key, 2)
    biases = jax.random.normal(subkey, (len(nodes), n_categories))
    bias_factor = CategoricalEBMFactor([Block(nodes)], biases)

    key, subkey = jax.random.split(key, 2)
    weights = jax.random.normal(subkey, (len(nodes) - 1, n_categories, n_categories))
    pair_factor = CategoricalEBMFactor([Block(nodes[:-1]), Block(nodes[1:])], weights)

    key, subkey = jax.random.split(key, 2)
    triplet_weights = jax.random.normal(subkey, (1, n_categories, n_categories, n_categories))
    triplet_factor = SquareCategoricalEBMFactor(
        [Block([nodes[2]]), Block([nodes[3]]), Block([extra_node])],
        triplet_weights,
    )

    _ = FactorizedEBM([bias_factor, pair_factor, triplet_factor])
    free_blocks = [Block(nodes[1::2]), Block(nodes[2::2])]
    clamped_blocks = [Block([nodes[0], extra_node])]
    gibbs_spec = BlockGibbsSpec(free_blocks, clamped_blocks)
    sampler = CategoricalGibbsConditional(n_categories)
    program = FactorSamplingProgram(
        gibbs_spec,
        [sampler, sampler],
        [bias_factor, pair_factor, triplet_factor],
        [],
    )

    block_keys = jax.random.split(key, len(free_blocks))
    init_state_free = [
        jax.random.randint(
            block_key,
            shape=(len(block.nodes),),
            minval=0,
            maxval=n_categories,
            dtype=jnp.uint8,
        )
        for block_key, block in zip(block_keys, free_blocks, strict=True)
    ]
    state_clamp = [jnp.array([1, 1], dtype=jnp.uint8)]
    return nodes, program, init_state_free, state_clamp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--parameter-kernels",
        choices=("ttmlir",),
        default="ttmlir",
    )
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--n-categories", type=int, default=3)
    parser.add_argument("--n-warmup", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=48)
    parser.add_argument("--steps-per-sample", type=int, default=3)
    parser.add_argument(
        "--system-desc-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=default_artifact_root(DEMO_NAME),
    )
    args = parser.parse_args()

    build_key = jax.random.key(args.seed)
    nodes, program, init_state_free, state_clamp = _build_program(
        key=build_key,
        n_categories=args.n_categories,
    )
    schedule = SamplingSchedule(
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        steps_per_sample=args.steps_per_sample,
    )
    sample_root = jax.random.fold_in(build_key, 17)
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
            samples = tt_thrml.sample_states(
                key,
                program,
                schedule,
                init_state_free,
                state_clamp,
                nodes_to_sample,
                backend=backend,
            )[0]
            elapsed = time.perf_counter() - start
            samples_np = np.asarray(samples, dtype=np.int64)
            runs.append(
                {
                    "run_index": run_index,
                    "elapsed_seconds": elapsed,
                    "samples_shape": list(samples_np.shape),
                    "samples_per_second": float(samples_np.shape[0] / elapsed),
                    "per_node_histograms": [
                        np.bincount(samples_np[:, i], minlength=args.n_categories).tolist()
                        for i in range(samples_np.shape[1])
                    ],
                }
            )

        print(
            json.dumps(
                {
                    "demo": "categorical_ebm",
                    "backend": args.parameter_kernels,
                    "device_id": args.device_id,
                    "n_categories": args.n_categories,
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
