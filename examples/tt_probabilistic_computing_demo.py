"""Port of upstream examples/00_probabilistic_computing.ipynb for TT execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.pgm import CategoricalNode
import ttnn

import tt_thrml
from _tt_demo_common import default_artifact_root, make_backend

DEMO_NAME = "probabilistic-computing"


def _build_program(*, side_length: int, n_categories: int, beta: float):
    graph = nx.grid_graph(dim=(side_length, side_length), periodic=False)
    coord_to_node = {coord: CategoricalNode() for coord in graph.nodes}
    nx.relabel_nodes(graph, coord_to_node, copy=False)
    for coord, node in coord_to_node.items():
        graph.nodes[node]["coords"] = coord

    bicol = nx.bipartite.color(graph)
    color0 = [node for node, color in bicol.items() if color == 0]
    color1 = [node for node, color in bicol.items() if color == 1]
    u_nodes, v_nodes = map(list, zip(*graph.edges()))

    id_mat = jnp.eye(n_categories, dtype=jnp.float32)
    weights = beta * jnp.broadcast_to(
        jnp.expand_dims(id_mat, 0),
        (len(u_nodes), n_categories, n_categories),
    )
    coupling_interaction = CategoricalEBMFactor([Block(u_nodes), Block(v_nodes)], weights)

    blocks = [Block(color0), Block(color1)]
    spec = BlockGibbsSpec(blocks, [])
    sampler = CategoricalGibbsConditional(n_categories)
    program = FactorSamplingProgram(
        spec,
        [sampler for _ in spec.free_blocks],
        [coupling_interaction],
        [],
    )
    return graph, spec, program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--parameter-kernels", choices=("ttmlir",), default="ttmlir")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--side-length", type=int, default=8)
    parser.add_argument("--n-categories", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--n-batches", type=int, default=16)
    parser.add_argument("--n-warmup", type=int, default=0)
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

    graph, spec, program = _build_program(
        side_length=args.side_length,
        n_categories=args.n_categories,
        beta=args.beta,
    )

    base_key = jax.random.key(args.seed)
    init_key, sample_root = jax.random.split(base_key)
    init_job_keys = jax.random.split(init_key, args.n_batches)
    init_states = []
    for job_key in init_job_keys:
        block_keys = jax.random.split(job_key, len(spec.free_blocks))
        init_states.append(
            [
                jax.random.randint(
                    block_key,
                    (len(block.nodes),),
                    minval=0,
                    maxval=args.n_categories,
                    dtype=jnp.uint8,
                )
                for block_key, block in zip(block_keys, spec.free_blocks, strict=True)
            ]
        )

    schedule = SamplingSchedule(
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        steps_per_sample=args.steps_per_sample,
    )
    nodes_to_sample = [Block(list(graph.nodes))]

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
            keys = jax.random.split(jax.random.fold_in(sample_root, run_index), args.n_batches)
            start = time.perf_counter()
            samples_many = tt_thrml.sample_states_many(
                keys,
                program,
                schedule,
                init_states,
                [[] for _ in range(args.n_batches)],
                nodes_to_sample,
                backend=backend,
            )
            elapsed = time.perf_counter() - start
            samples_np = np.stack(
                [np.asarray(job_samples[0], dtype=np.int64) for job_samples in samples_many],
                axis=0,
            )
            flat_samples = samples_np.reshape(-1, samples_np.shape[-1])
            runs.append(
                {
                    "run_index": run_index,
                    "elapsed_seconds": elapsed,
                    "samples_shape": list(samples_np.shape),
                    "samples_per_second": float(flat_samples.shape[0] / elapsed),
                    "per_node_histograms": [
                        np.bincount(flat_samples[:, i], minlength=args.n_categories).tolist()
                        for i in range(flat_samples.shape[1])
                    ],
                }
            )

        print(
            json.dumps(
                {
                    "demo": "probabilistic_computing",
                    "backend": args.parameter_kernels,
                    "grid_side_length": args.side_length,
                    "n_categories": args.n_categories,
                    "n_batches": args.n_batches,
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
