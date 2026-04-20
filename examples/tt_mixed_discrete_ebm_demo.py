"""Run an upstream-style mixed spin/categorical discrete EBM on TT hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    DiscreteEBMFactor,
    SpinEBMFactor,
    SpinGibbsConditional,
    SquareDiscreteEBMFactor,
)
from thrml.pgm import CategoricalNode, SpinNode
import ttnn

import tt_thrml
from _tt_demo_common import default_artifact_root, make_backend

DEMO_NAME = "mixed-discrete-ebm"


def _build_program(*, key, n_categories: int):
    bin_nodes = [SpinNode() for _ in range(3)]
    cat_nodes = [CategoricalNode() for _ in range(4)]

    key, subkey = jax.random.split(key, 2)
    bin_bias_fac = SpinEBMFactor(
        [Block(bin_nodes)],
        jax.random.normal(subkey, (len(bin_nodes),)),
    )

    key, subkey = jax.random.split(key, 2)
    cat_bias_fac = CategoricalEBMFactor(
        [Block(cat_nodes)],
        jax.random.normal(subkey, (len(cat_nodes), n_categories)),
    )

    key, subkey = jax.random.split(key, 2)
    weight_fac = DiscreteEBMFactor(
        [Block(bin_nodes)],
        [Block(cat_nodes[:-1])],
        jax.random.normal(subkey, (len(bin_nodes), n_categories)),
    )

    key, subkey = jax.random.split(key, 2)
    triple_weight_fac = SquareDiscreteEBMFactor(
        [Block([bin_nodes[-1]]), Block([bin_nodes[-2]])],
        [Block([cat_nodes[-1]])],
        jax.random.normal(subkey, (1, n_categories)),
    )

    free_blocks = [
        Block(bin_nodes[1::2]),
        Block(cat_nodes[0:-1:2]),
        Block(bin_nodes[::2]),
        Block(cat_nodes[1:-1:2]),
        Block([cat_nodes[-1]]),
    ]
    program = FactorSamplingProgram(
        BlockGibbsSpec(free_blocks, []),
        [
            SpinGibbsConditional(),
            CategoricalGibbsConditional(n_categories),
            SpinGibbsConditional(),
            CategoricalGibbsConditional(n_categories),
            CategoricalGibbsConditional(n_categories),
        ],
        [bin_bias_fac, cat_bias_fac, weight_fac, triple_weight_fac],
        [],
    )

    init_keys = jax.random.split(key, len(free_blocks) + 1)
    init_state_free = [
        jax.random.bernoulli(init_keys[0], 0.5, (1,)).astype(jnp.bool_),
        jax.random.randint(
            init_keys[1],
            (2,),
            minval=0,
            maxval=n_categories,
            dtype=jnp.uint8,
        ),
        jax.random.bernoulli(init_keys[2], 0.5, (2,)).astype(jnp.bool_),
        jax.random.randint(
            init_keys[3],
            (1,),
            minval=0,
            maxval=n_categories,
            dtype=jnp.uint8,
        ),
        jax.random.randint(
            init_keys[4],
            (1,),
            minval=0,
            maxval=n_categories,
            dtype=jnp.uint8,
        ),
    ]

    return (
        bin_nodes,
        cat_nodes,
        program,
        init_state_free,
        [],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--parameter-kernels",
        choices=("ttmlir",),
        default="ttmlir",
    )
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--n-categories", type=int, default=3)
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
    bin_nodes, cat_nodes, program, init_state_free, state_clamp = _build_program(
        key=build_key,
        n_categories=args.n_categories,
    )
    schedule = SamplingSchedule(
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        steps_per_sample=args.steps_per_sample,
    )
    sample_root = jax.random.fold_in(build_key, 23)
    nodes_to_sample = [Block(bin_nodes), Block(cat_nodes)]

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
            spin_samples = np.asarray(outputs[0])
            categorical_samples = np.asarray(outputs[1], dtype=np.int64)
            runs.append(
                {
                    "run_index": run_index,
                    "elapsed_seconds": elapsed,
                    "spin_shape": list(spin_samples.shape),
                    "categorical_shape": list(categorical_samples.shape),
                    "samples_per_second": float(spin_samples.shape[0] / elapsed),
                    "spin_mean": float(np.where(spin_samples, 1.0, -1.0).mean()),
                    "categorical_histograms": [
                        np.bincount(
                            categorical_samples[:, i],
                            minlength=args.n_categories,
                        ).tolist()
                        for i in range(categorical_samples.shape[1])
                    ],
                }
            )

        print(
            json.dumps(
                {
                    "demo": "mixed_discrete_ebm",
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
