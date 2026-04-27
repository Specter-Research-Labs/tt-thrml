#!/usr/bin/env python3
"""Test tt_thrml fused executor with a simple Ising chain on Tenstorrent hardware."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import jax
import numpy as np
import thrml
import ttnn
from jax import numpy as jnp
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.observers import MomentAccumulatorObserver

sys.path.insert(0, str(Path(__file__).parent.parent))

import tt_thrml


def build_ising_chain():
    """Build a simple 5-node Ising chain."""
    nodes = [thrml.SpinNode() for _ in range(5)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
    biases = jnp.zeros((5,), dtype=jnp.float32)
    weights = jnp.ones((4,), dtype=jnp.float32) * 0.5
    beta = jnp.array(1.0, dtype=jnp.float32)
    model = IsingEBM(nodes, edges, biases, weights, beta)
    free_blocks = [thrml.Block(nodes[::2]), thrml.Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    return nodes, model, free_blocks, program


def main():
    print("=" * 60)
    print("tt_thrml Test: Fused Executor + Bulk RNG")
    print("=" * 60)

    system_desc_path = os.environ.get("SYSTEM_DESC_PATH")
    if not system_desc_path:
        print("ERROR: SYSTEM_DESC_PATH not set")
        sys.exit(1)

    build_dir = os.environ.get("TTMLIR_BUILD_DIR")
    artifact_root = Path(os.environ.get("SPECTER_ARTIFACT_ROOT", "/tmp")) / "tt-thrml-test"

    print(f"System desc: {system_desc_path}")
    print(f"Build dir: {build_dir}")
    print(f"Artifact root: {artifact_root}")
    print()

    nodes, model, free_blocks, program = build_ising_chain()
    print(f"Built Ising chain with {len(nodes)} nodes, {len(free_blocks)} free blocks")

    schedule = thrml.SamplingSchedule(
        n_warmup=8,
        n_samples=16,
        steps_per_sample=2,
    )

    key = jax.random.key(42)
    init_key, sample_key = jax.random.split(key)
    init_state_free = hinton_init(init_key, model, free_blocks, ())
    state_clamp = []
    nodes_to_sample = free_blocks

    print("\n--- Opening device ---")
    device = tt_thrml.open_device(ttnn, device_id=0)
    print(f"Device opened: {type(device)}")

    try:
        print("\n--- Creating TT-MLIR config ---")
        config = tt_thrml.make_ttmlir_config(
            system_desc_path=system_desc_path,
            artifact_root=artifact_root,
            build_dir=build_dir,
        )
        print(f"Config created: {config}")

        total_sweeps = schedule.n_warmup + schedule.n_samples * schedule.steps_per_sample
        print(f"\n--- Creating fused executor (n_sweeps={total_sweeps}) ---")
        executor = tt_thrml.make_executor(ttnn, device, program, config, n_sweeps=total_sweeps)
        print(f"Executor created with {len(executor.compiled.blocks)} fused blocks")

        for i, block in enumerate(executor.compiled.blocks):
            spec = block.spec
            n_terms = sum(ispec.n_terms for ispec in spec.interactions)
            print(
                f"  Block {i}: family={spec.family.value}, n_nodes={spec.n_nodes}, "
                f"n_interactions={len(spec.interactions)}, n_terms={n_terms}, "
                f"global_start={spec.block_global_start}"
            )

        print("\n--- Loading state ---")
        executor.load_state(init_state_free, state_clamp)
        print("State loaded to device")

        print("\n--- Preparing bulk RNG ---")
        start = time.perf_counter()
        executor.prepare_rng(sample_key)
        rng_time = time.perf_counter() - start
        print(f"RNG prepared: {total_sweeps} sweeps in {rng_time:.3f}s")

        print("\n--- Running warmup (no host RNG!) ---")
        start = time.perf_counter()
        executor.run_warmup(schedule.n_warmup)
        warmup_time = time.perf_counter() - start
        print(f"Warmup done: {schedule.n_warmup} sweeps in {warmup_time:.3f}s")

        print("\n--- Sampling (no host RNG!) ---")
        samples = []
        start = time.perf_counter()
        for i in range(schedule.n_samples):
            for _ in range(schedule.steps_per_sample):
                executor.run_sweep()
            obs = executor.observe(nodes_to_sample)
            samples.append(obs)
        sample_time = time.perf_counter() - start
        print(f"Sampling done: {schedule.n_samples} samples in {sample_time:.3f}s")

        print("\n--- Results ---")
        per_block_samples = {block: [] for block in nodes_to_sample}
        for obs in samples:
            for block, state in obs.items():
                per_block_samples[block].append(np.asarray(state))

        for block_idx, (block, block_samples) in enumerate(per_block_samples.items()):
            if not block_samples:
                continue
            samples_arr = np.stack(block_samples)
            signed = np.where(samples_arr > 0, 1.0, -1.0)
            print(f"Block {block_idx}: shape={samples_arr.shape}, mean={signed.mean():.4f}")

        print(f"Samples/sec: {schedule.n_samples / sample_time:.1f}")
        print(
            f"Sweeps/sec: {(schedule.n_warmup + schedule.n_samples * schedule.steps_per_sample) / (warmup_time + sample_time):.1f}"
        )
        sys.stdout.flush()

        print("\n--- sample_states (StateObserver via high-level API) ---")
        sys.stdout.flush()
        obs_key = jax.random.fold_in(sample_key, 99)
        start = time.perf_counter()
        state_results = executor.sample_states(
            obs_key,
            schedule,
            nodes_to_sample,
            init_state_free=init_state_free,
            state_clamp=state_clamp,
        )
        obs_time = time.perf_counter() - start
        print(f"sample_states done in {obs_time:.3f}s")
        for block_idx, (block, block_states) in enumerate(zip(nodes_to_sample, state_results)):
            arr = np.asarray(block_states)
            signed = np.where(arr > 0, 1.0, -1.0)
            print(f"  Block {block_idx}: shape={arr.shape}, mean={signed.mean():.4f}")
        assert len(state_results) == len(nodes_to_sample), "wrong number of result blocks"
        assert state_results[0].shape[0] == schedule.n_samples, "wrong sample count"
        sys.stdout.flush()

        print("\n--- sample_with_observation (MomentAccumulatorObserver) ---")
        sys.stdout.flush()
        moment_spec = [
            [(n,) for n in nodes],
            [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))],
        ]
        observer = MomentAccumulatorObserver(moment_spec)
        start = time.perf_counter()
        carry, obs_out = executor.sample_with_observation(
            obs_key,
            schedule,
            observer,
            init_state_free=init_state_free,
            state_clamp=state_clamp,
        )
        moment_time = time.perf_counter() - start
        print(f"MomentAccumulatorObserver done in {moment_time:.3f}s")
        first_moments, second_moments = carry
        first_means = np.asarray(first_moments) / schedule.n_samples
        second_means = np.asarray(second_moments) / schedule.n_samples
        assert obs_out is None, "MomentAccumulatorObserver should return None observations"
        assert first_means.shape == (len(nodes),), f"unexpected first moments shape: {first_means.shape}"
        assert second_means.shape == (
            len(nodes) * (len(nodes) - 1) // 2,
        ), f"unexpected second moments shape: {second_means.shape}"
        print(f"  First moments (mean spin):  {np.round(first_means, 4)}")
        print(f"  Second moments (mean corr): {np.round(second_means, 4)}")
        sys.stdout.flush()

        print("\n" + "=" * 60)
        print("SUCCESS: tt_thrml fused executor test passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n!!! ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        print("\n--- Closing device ---")
        tt_thrml.close_device(ttnn, device)
        print("Device closed")


if __name__ == "__main__":
    main()
