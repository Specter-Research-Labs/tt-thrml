"""Run a small upstream-style Ising chain on Tenstorrent hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
from jax import numpy as jnp
import numpy as np
import thrml
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
import ttnn

import tt_thrml
from _tt_demo_common import default_artifact_root, make_backend

DEMO_NAME = "ising-chain"


def _build_program():
    nodes = [thrml.SpinNode() for _ in range(5)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
    biases = jnp.zeros((5,), dtype=jnp.float32)
    weights = jnp.ones((4,), dtype=jnp.float32) * 0.5
    beta = jnp.array(1.0, dtype=jnp.float32)
    model = IsingEBM(nodes, edges, biases, weights, beta)
    free_blocks = [thrml.Block(nodes[::2]), thrml.Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    return nodes, model, free_blocks, program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--parameter-kernels",
        choices=("native", "ttmlir"),
        default="ttmlir",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--n-warmup", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--steps-per-sample", type=int, default=2)
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

    nodes, model, free_blocks, program = _build_program()
    schedule = thrml.SamplingSchedule(
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        steps_per_sample=args.steps_per_sample,
    )
    base_key = jax.random.key(args.seed)
    init_key, sample_root = jax.random.split(base_key, 2)
    init_state_free = hinton_init(init_key, model, free_blocks, ())
    state_clamp = []
    nodes_to_sample = [thrml.Block(nodes)]

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
            samples_np = np.asarray(samples)
            signed = np.where(samples_np, 1.0, -1.0)
            runs.append(
                {
                    "run_index": run_index,
                    "elapsed_seconds": elapsed,
                    "samples_shape": list(samples_np.shape),
                    "mean_spin": float(signed.mean()),
                    "per_node_mean_spin": signed.mean(axis=0).tolist(),
                    "samples_per_second": float(samples_np.shape[0] / elapsed),
                }
            )

        print(
            json.dumps(
                {
                    "demo": "ising_chain",
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
