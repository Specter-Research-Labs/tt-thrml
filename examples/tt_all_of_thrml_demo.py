"""Sampling-focused port of upstream examples/01_all_of_thrml.ipynb."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule
from thrml.factor import FactorSamplingProgram
import ttnn

import tt_thrml
from _tt_demo_common import default_artifact_root, make_backend
from _tt_upstream_example_common import (
    ContinuousNode,
    CouplingFactor,
    ExtendedSpinGibbsSampler,
    GaussianSampler,
    LinearFactor,
    QuadraticFactor,
    SpinNode,
    build_skip_graph_from_grid,
    construct_inv_cov,
    continuous_moment_observer,
    generate_continuous_grid_graph,
    make_continuous_block_spec,
    make_mixed_block_spec,
    make_random_typed_grid,
    partition_edges_by_type,
)
from thrml.models.discrete_ebm import SpinEBMFactor

DEMO_NAME = "all-of-thrml"


def _build_gaussian_program(*, key, side_length: int):
    colors, edges, _graph = generate_continuous_grid_graph(side_length, side_length)
    all_nodes = colors[1] + colors[2]
    node_map = dict(zip(all_nodes, range(len(all_nodes)), strict=True))

    key, subkey = jax.random.split(key, 2)
    cov_inv_diag = jax.random.uniform(
        subkey,
        (len(all_nodes),),
        minval=1.0,
        maxval=2.0,
    )
    key, subkey = jax.random.split(key, 2)
    cov_inv_off_diag = jax.random.uniform(
        subkey,
        (len(edges[0]),),
        minval=-0.25,
        maxval=0.25,
    )
    inv_cov_mat = construct_inv_cov(
        diag=cov_inv_diag,
        all_edges=edges,
        off_diag=cov_inv_off_diag,
        node_map=node_map,
    )
    key, subkey = jax.random.split(key, 2)
    mean_vec = jax.random.normal(subkey, (len(all_nodes),))
    bias_vec = -1.0 * jnp.einsum("ij, i -> j", inv_cov_mat, mean_vec)

    spec = make_continuous_block_spec(color0=colors[1], color1=colors[2])
    sampler = GaussianSampler()
    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler, sampler],
        factors=[
            LinearFactor(bias_vec, Block(all_nodes)),
            QuadraticFactor(1.0 / cov_inv_diag, Block(all_nodes)),
            CouplingFactor(cov_inv_off_diag, (Block(edges[0]), Block(edges[1]))),
        ],
        other_interaction_groups=[],
    )
    observer = continuous_moment_observer(edges=edges)
    return all_nodes, edges, inv_cov_mat, spec, program, observer


def _build_mixed_program(*, key, rows: int, cols: int, skip_lengths: tuple[int, ...]):
    grid, coloring = make_random_typed_grid(rows=rows, cols=cols, seed=int(jax.random.randint(key, (), 0, 1_000_000)))
    edges, _graph = build_skip_graph_from_grid(grid, list(skip_lengths))
    spin_nodes, cont_nodes, ss_edges, cc_edges, sc_edges = partition_edges_by_type(edges)

    key, subkey = jax.random.split(key, 2)
    cont_quad = QuadraticFactor(
        jax.random.uniform(subkey, (len(cont_nodes),), minval=2.0, maxval=3.0),
        Block(cont_nodes),
    )
    key, subkey = jax.random.split(key, 2)
    cont_linear = LinearFactor(jax.random.normal(subkey, (len(cont_nodes),)), Block(cont_nodes))
    key, subkey = jax.random.split(key, 2)
    cont_coupling = CouplingFactor(
        jax.random.uniform(subkey, (len(cc_edges[0]),), minval=-0.1, maxval=0.1),
        (Block(cc_edges[0]), Block(cc_edges[1])),
    )
    key, subkey = jax.random.split(key, 2)
    spin_con_coupling = CouplingFactor(
        jax.random.normal(subkey, (len(sc_edges[0]),)),
        (Block(sc_edges[0]), Block(sc_edges[1])),
    )
    key, subkey = jax.random.split(key, 2)
    spin_linear = SpinEBMFactor([Block(spin_nodes)], jax.random.normal(subkey, (len(spin_nodes),)))
    key, subkey = jax.random.split(key, 2)
    spin_coupling = SpinEBMFactor(
        [Block(ss_edges[0]), Block(ss_edges[1])],
        jax.random.normal(subkey, (len(ss_edges[0]),)),
    )

    block_spec = make_mixed_block_spec(coloring_by_type=coloring)
    gaussian_sampler = GaussianSampler()
    spin_sampler = ExtendedSpinGibbsSampler()
    samplers = []
    for block in block_spec.free_blocks:
        if isinstance(block.nodes[0], SpinNode):
            samplers.append(spin_sampler)
        else:
            samplers.append(gaussian_sampler)

    program = FactorSamplingProgram(
        block_spec,
        samplers,
        [cont_quad, cont_linear, cont_coupling, spin_con_coupling, spin_linear, spin_coupling],
        [],
    )
    return spin_nodes, cont_nodes, block_spec, program


def _init_gaussian_state(*, key, free_blocks):
    block_keys = jax.random.split(key, len(free_blocks))
    return [
        0.1 * jax.random.normal(block_key, (len(block.nodes),), dtype=jnp.float32)
        for block_key, block in zip(block_keys, free_blocks, strict=True)
    ]


def _init_mixed_state(*, key, free_blocks):
    states = []
    split_keys = jax.random.split(key, len(free_blocks))
    for block_key, block in zip(split_keys, free_blocks, strict=True):
        init_shape = (len(block.nodes),)
        if isinstance(block.nodes[0], ContinuousNode):
            states.append(0.1 * jax.random.normal(block_key, init_shape))
        else:
            states.append(jax.random.bernoulli(block_key, 0.5, init_shape))
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--parameter-kernels", choices=("ttmlir",), default="ttmlir")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--gaussian-side-length", type=int, default=5)
    parser.add_argument("--gaussian-batch-size", type=int, default=8)
    parser.add_argument("--gaussian-n-warmup", type=int, default=8)
    parser.add_argument("--gaussian-n-samples", type=int, default=24)
    parser.add_argument("--gaussian-steps-per-sample", type=int, default=5)
    parser.add_argument("--mixed-rows", type=int, default=8)
    parser.add_argument("--mixed-cols", type=int, default=8)
    parser.add_argument("--mixed-batch-size", type=int, default=8)
    parser.add_argument("--mixed-n-warmup", type=int, default=16)
    parser.add_argument("--mixed-n-samples", type=int, default=24)
    parser.add_argument("--mixed-steps-per-sample", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--system-desc-path", type=Path, default=None)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=default_artifact_root(DEMO_NAME),
    )
    args = parser.parse_args()

    base_key = jax.random.key(args.seed)
    gaussian_build_key, mixed_build_key, init_key, sample_root = jax.random.split(base_key, 4)
    gaussian_nodes, gaussian_edges, inv_cov_mat, gaussian_spec, gaussian_program, observer = _build_gaussian_program(
        key=gaussian_build_key,
        side_length=args.gaussian_side_length,
    )
    spin_nodes, cont_nodes, mixed_spec, mixed_program = _build_mixed_program(
        key=mixed_build_key,
        rows=args.mixed_rows,
        cols=args.mixed_cols,
        skip_lengths=(1, 3, 5),
    )

    gaussian_init_key, mixed_init_key = jax.random.split(init_key)
    gaussian_init_states = [
        _init_gaussian_state(key=job_key, free_blocks=gaussian_spec.free_blocks)
        for job_key in jax.random.split(gaussian_init_key, args.gaussian_batch_size)
    ]
    mixed_init_states = [
        _init_mixed_state(key=job_key, free_blocks=mixed_spec.free_blocks)
        for job_key in jax.random.split(mixed_init_key, args.mixed_batch_size)
    ]

    gaussian_schedule = SamplingSchedule(
        n_warmup=args.gaussian_n_warmup,
        n_samples=args.gaussian_n_samples,
        steps_per_sample=args.gaussian_steps_per_sample,
    )
    mixed_schedule = SamplingSchedule(
        n_warmup=args.mixed_n_warmup,
        n_samples=args.mixed_n_samples,
        steps_per_sample=args.mixed_steps_per_sample,
    )

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
            gaussian_keys = jax.random.split(
                jax.random.fold_in(sample_root, run_index * 2),
                args.gaussian_batch_size,
            )
            mixed_keys = jax.random.split(
                jax.random.fold_in(sample_root, run_index * 2 + 1),
                args.mixed_batch_size,
            )

            start = time.perf_counter()
            gaussian_results_many = tt_thrml.sample_with_observation_many(
                gaussian_keys,
                gaussian_program,
                gaussian_schedule,
                gaussian_init_states,
                [[] for _ in range(args.gaussian_batch_size)],
                [observer.init() for _ in range(args.gaussian_batch_size)],
                observer,
                backend=backend,
            )
            gaussian_moments_many = [
                moments for moments, _ in gaussian_results_many
            ]
            gaussian_elapsed = time.perf_counter() - start

            gaussian_moments = jax.tree.map(
                lambda *values: np.mean(
                    np.stack([np.asarray(value) for value in values], axis=0),
                    axis=0,
                )
                / gaussian_schedule.n_samples,
                *gaussian_moments_many,
            )
            gaussian_covariances = gaussian_moments[-1] - (
                gaussian_moments[0] * gaussian_moments[1]
            )
            cov = np.linalg.inv(np.asarray(inv_cov_mat))
            node_map = dict(zip(gaussian_nodes, range(len(gaussian_nodes)), strict=True))
            real_covariances = np.array(
                [
                    cov[node_map[lhs], node_map[rhs]]
                    for lhs, rhs in zip(*gaussian_edges, strict=True)
                ],
                dtype=np.float32,
            )
            gaussian_error = float(
                np.max(np.abs(real_covariances - gaussian_covariances))
                / np.abs(np.max(real_covariances))
            )

            start = time.perf_counter()
            mixed_samples_many = tt_thrml.sample_states_many(
                mixed_keys,
                mixed_program,
                mixed_schedule,
                mixed_init_states,
                [[] for _ in range(args.mixed_batch_size)],
                [Block(spin_nodes), Block(cont_nodes)],
                backend=backend,
            )
            mixed_elapsed = time.perf_counter() - start

            spin_samples = np.stack(
                [np.asarray(job_samples[0], dtype=bool) for job_samples in mixed_samples_many],
                axis=0,
            ).reshape(-1, len(spin_nodes))
            cont_samples = np.stack(
                [np.asarray(job_samples[1], dtype=np.float32) for job_samples in mixed_samples_many],
                axis=0,
            ).reshape(-1, len(cont_nodes))
            signed_spin = np.where(spin_samples, 1.0, -1.0)

            runs.append(
                {
                    "run_index": run_index,
                    "gaussian": {
                        "elapsed_seconds": gaussian_elapsed,
                        "max_relative_covariance_error": gaussian_error,
                        "mean_first_moment": float(np.mean(gaussian_moments[0])),
                    },
                    "mixed": {
                        "elapsed_seconds": mixed_elapsed,
                        "spin_sample_shape": list(np.asarray(mixed_samples_many[0][0]).shape),
                        "continuous_sample_shape": list(np.asarray(mixed_samples_many[0][1]).shape),
                        "spin_mean": float(signed_spin.mean()),
                        "continuous_mean": cont_samples.mean(axis=0).tolist(),
                        "continuous_std": cont_samples.std(axis=0).tolist(),
                    },
                }
            )

        print(
            json.dumps(
                {
                    "demo": "all_of_thrml",
                    "backend": args.parameter_kernels,
                    "gaussian_grid_side_length": args.gaussian_side_length,
                    "gaussian_batch_size": args.gaussian_batch_size,
                    "mixed_grid": [args.mixed_rows, args.mixed_cols],
                    "mixed_batch_size": args.mixed_batch_size,
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
