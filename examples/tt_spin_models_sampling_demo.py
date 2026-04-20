"""Sampling-focused port of upstream examples/02_spin_models.ipynb."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import dwave_networkx
import jax
import jax.numpy as jnp
import numpy as np
from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.pgm import SpinNode
import networkx as nx
import ttnn

import tt_thrml
from _tt_demo_common import default_artifact_root, make_backend

DEMO_NAME = "spin-models-sampling"


def _build_program(*, key, pegasus_size: int):
    graph = dwave_networkx.pegasus_graph(pegasus_size)
    coord_to_node = {coord: SpinNode() for coord in graph.nodes}
    nx.relabel_nodes(graph, coord_to_node, copy=False)

    nodes = list(graph.nodes)
    edges = list(graph.edges)

    key, subkey = jax.random.split(key, 2)
    biases = jax.random.normal(subkey, (len(nodes),), dtype=jnp.float32)
    key, subkey = jax.random.split(key, 2)
    weights = jax.random.normal(subkey, (len(edges),), dtype=jnp.float32)
    beta = jnp.array(1.0, dtype=jnp.float32)

    model = IsingEBM(nodes, edges, biases, weights, beta)

    coloring = nx.coloring.greedy_color(graph, strategy="DSATUR")
    n_colors = max(coloring.values()) + 1
    free_coloring = [[] for _ in range(n_colors)]
    for node in graph.nodes:
        free_coloring[coloring[node]].append(node)
    free_blocks = [Block(group) for group in free_coloring if group]

    program = IsingSamplingProgram(model, free_blocks, [])
    return nodes, model, free_blocks, program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--parameter-kernels", choices=("ttmlir",), default="ttmlir")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--pegasus-size", type=int, default=4)
    parser.add_argument("--n-chains", type=int, default=8)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--steps-per-sample", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--system-desc-path", type=Path, default=None)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=default_artifact_root(DEMO_NAME),
    )
    args = parser.parse_args()

    build_key = jax.random.key(args.seed)
    nodes, model, free_blocks, program = _build_program(
        key=build_key,
        pegasus_size=args.pegasus_size,
    )
    init_key, sample_root = jax.random.split(build_key)
    init_state_frees = [
        hinton_init(job_key, model, free_blocks, ())
        for job_key in jax.random.split(init_key, args.n_chains)
    ]
    schedule = SamplingSchedule(
        args.n_warmup,
        args.n_samples,
        args.steps_per_sample,
    )
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
            keys = jax.random.split(jax.random.fold_in(sample_root, run_index), args.n_chains)
            start = time.perf_counter()
            samples_many = tt_thrml.sample_states_many(
                keys,
                program,
                schedule,
                init_state_frees,
                [[] for _ in range(args.n_chains)],
                nodes_to_sample,
                backend=backend,
            )
            elapsed = time.perf_counter() - start
            samples_np = np.stack(
                [np.asarray(job_samples[0], dtype=bool) for job_samples in samples_many],
                axis=0,
            )
            flat_samples = samples_np.reshape(-1, samples_np.shape[-1])
            signed = np.where(flat_samples, 1.0, -1.0)
            runs.append(
                {
                    "run_index": run_index,
                    "elapsed_seconds": elapsed,
                    "samples_shape": list(samples_np.shape),
                    "samples_per_second": float(flat_samples.shape[0] / elapsed),
                    "mean_spin": float(signed.mean()),
                    "per_node_mean_spin": signed.mean(axis=0).tolist(),
                }
            )

        print(
            json.dumps(
                {
                    "demo": "spin_models_sampling",
                    "backend": args.parameter_kernels,
                    "pegasus_size": args.pegasus_size,
                    "n_chains": args.n_chains,
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
