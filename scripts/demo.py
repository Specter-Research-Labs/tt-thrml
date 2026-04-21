#!/usr/bin/env python3
"""
THRML demo: four programs, CPU (JAX) vs Wormhole (tt_thrml), with Tracy profiling.

Programs:
  1. Ising spin chain        -- ferromagnetic 1D chain, 8 nodes
  2. Categorical EBM         -- 6 nodes, 4 categories
  3. Gaussian chain          -- 6 continuous nodes (GaussianConditional)
  4. Mixed                   -- 3 spin + 3 categorical + 3 Gaussian

Usage:
  python3 scripts/demo.py
  TT_DEMO_CPU_ONLY=1 python3 scripts/demo.py       # skip Wormhole, no ttnn needed

  With Tracy profiling (requires TT environment):
  python3 -m tracy -r -n thrml-demo scripts/demo.py
  TT_METAL_DEVICE_PROFILER=1 python3 -m tracy -r scripts/demo.py

Required env (Wormhole path):
  SYSTEM_DESC_PATH   path to system_desc.ttsys
  TTMLIR_BUILD_DIR   path to TT-MLIR build directory

Optional env:
  TT_DEMO_CPU_ONLY   set to 1 to skip Wormhole
  TT_DEMO_PROFILE    set to 1 to print per-family dispatch vs kernel timing
  TT_DEMO_DEVICE_ID  Wormhole device ID (default: 0)
  TT_DEMO_ARTIFACT_ROOT  artifact cache root (default: /tmp/tt-thrml-demo)
"""

from __future__ import annotations

import os
import sys
import time
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
    sample_states as cpu_sample_states,
)
from thrml.factor import AbstractFactor, FactorSamplingProgram
from thrml.interaction import InteractionGroup
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    SpinEBMFactor,
    SpinGibbsConditional,
)
from thrml.pgm import AbstractNode, CategoricalNode, SpinNode

import tt_thrml


# ── Custom node + interaction types (Gaussian family) ─────────────────────────

class ContinuousNode(AbstractNode):
    pass


class _LinearInteraction(eqx.Module):
    weights: jax.Array


class _QuadraticInteraction(eqx.Module):
    inverse_weights: jax.Array


def _linear_factor(weights: jax.Array, block: Block) -> AbstractFactor:
    class _F(AbstractFactor):
        _w: jax.Array
        def __init__(self):
            super().__init__([block])
            self._w = weights
        def to_interaction_groups(self):
            return [InteractionGroup(_LinearInteraction(self._w), self.node_groups[0], [])]
    return _F()


def _quadratic_factor(inverse_weights: jax.Array, block: Block) -> AbstractFactor:
    class _F(AbstractFactor):
        _w: jax.Array
        def __init__(self):
            super().__init__([block])
            self._w = inverse_weights
        def to_interaction_groups(self):
            return [InteractionGroup(_QuadraticInteraction(self._w), self.node_groups[0], [])]
    return _F()


def _coupling_factor(weights: jax.Array, block_a: Block, block_b: Block) -> AbstractFactor:
    class _F(AbstractFactor):
        _w: jax.Array
        def __init__(self):
            super().__init__([block_a, block_b])
            self._w = weights
        def to_interaction_groups(self):
            return [
                InteractionGroup(_LinearInteraction(self._w), self.node_groups[0], [self.node_groups[1]]),
                InteractionGroup(_LinearInteraction(self._w), self.node_groups[1], [self.node_groups[0]]),
            ]
    return _F()


# ── Demo programs ─────────────────────────────────────────────────────────────

SCHEDULE = SamplingSchedule(n_warmup=64, n_samples=256, steps_per_sample=4)


@dataclass(frozen=True)
class Demo:
    name: str
    program: FactorSamplingProgram
    schedule: SamplingSchedule
    init_state_free: list
    state_clamp: list
    nodes_to_sample: list[Block]


def _ising_chain() -> Demo:
    n = 8
    nodes = [SpinNode() for _ in range(n)]
    blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    bias = jnp.array([0.3, -0.2, 0.1, -0.15, 0.25, -0.1, 0.2, -0.05], dtype=jnp.float32)
    program = FactorSamplingProgram(
        BlockGibbsSpec(blocks, []),
        [SpinGibbsConditional(), SpinGibbsConditional()],
        [
            SpinEBMFactor([Block(nodes)], bias),
            SpinEBMFactor([Block(nodes[:-1]), Block(nodes[1:])], jnp.full(n - 1, 0.5, dtype=jnp.float32)),
        ],
        [],
    )
    keys = jax.random.split(jax.random.key(101), len(blocks))
    return Demo(
        name="Ising chain",
        program=program,
        schedule=SCHEDULE,
        init_state_free=[jax.random.bernoulli(k, 0.5, (len(b.nodes),)).astype(jnp.bool_) for k, b in zip(keys, blocks)],
        state_clamp=[],
        nodes_to_sample=[Block(nodes)],
    )


def _categorical_ebm() -> Demo:
    n, c = 6, 4
    nodes = [CategoricalNode() for _ in range(n)]
    blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    bias = jnp.zeros((n, c), dtype=jnp.float32).at[:, 0].set(0.3).at[:, 1].set(-0.1)
    pair = (jnp.eye(c, dtype=jnp.float32) * 0.4)[None].repeat(n - 1, axis=0)
    sampler = CategoricalGibbsConditional(c)
    program = FactorSamplingProgram(
        BlockGibbsSpec(blocks, []),
        [sampler, sampler],
        [
            CategoricalEBMFactor([Block(nodes)], bias),
            CategoricalEBMFactor([Block(nodes[:-1]), Block(nodes[1:])], pair),
        ],
        [],
    )
    keys = jax.random.split(jax.random.key(102), len(blocks))
    return Demo(
        name="Categorical EBM",
        program=program,
        schedule=SCHEDULE,
        init_state_free=[jax.random.randint(k, (len(b.nodes),), 0, c, dtype=jnp.uint8) for k, b in zip(keys, blocks)],
        state_clamp=[],
        nodes_to_sample=[Block(nodes)],
    )


def _gaussian_chain() -> Demo:
    n = 6
    nodes = [ContinuousNode() for _ in range(n)]
    blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    sdt = {ContinuousNode: jax.ShapeDtypeStruct((), jnp.float32)}
    all_nodes = Block(nodes)
    program = FactorSamplingProgram(
        BlockGibbsSpec(blocks, [], sdt),
        [tt_thrml.GaussianConditional(), tt_thrml.GaussianConditional()],
        [
            _linear_factor(jnp.zeros(n, dtype=jnp.float32), all_nodes),
            _quadratic_factor(jnp.full(n, 0.8, dtype=jnp.float32), all_nodes),
            _coupling_factor(jnp.full(n - 1, 0.15, dtype=jnp.float32), Block(nodes[:-1]), Block(nodes[1:])),
        ],
        [],
    )
    keys = jax.random.split(jax.random.key(103), len(blocks))
    return Demo(
        name="Gaussian chain",
        program=program,
        schedule=SCHEDULE,
        init_state_free=[jax.random.normal(k, (len(b.nodes),), dtype=jnp.float32) * 0.2 for k, b in zip(keys, blocks)],
        state_clamp=[],
        nodes_to_sample=[Block(nodes)],
    )


def _mixed_program() -> Demo:
    n, c = 3, 3
    spins = [SpinNode() for _ in range(n)]
    cats = [CategoricalNode() for _ in range(n)]
    gauss = [ContinuousNode() for _ in range(n)]
    blocks = [
        Block([spins[0]]), Block([cats[0]]), Block([gauss[0]]),
        Block([spins[1]]), Block([cats[1]]), Block([gauss[1]]),
        Block([spins[2]]), Block([cats[2]]), Block([gauss[2]]),
    ]
    sdt = {
        SpinNode: jax.ShapeDtypeStruct((), jnp.bool_),
        CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint8),
        ContinuousNode: jax.ShapeDtypeStruct((), jnp.float32),
    }
    cat_s = CategoricalGibbsConditional(c)
    gauss_block = Block(gauss)
    program = FactorSamplingProgram(
        BlockGibbsSpec(blocks, [], sdt),
        [SpinGibbsConditional(), cat_s, tt_thrml.GaussianConditional()] * n,
        [
            SpinEBMFactor([Block(spins)], jnp.array([0.2, -0.3, 0.1], dtype=jnp.float32)),
            CategoricalEBMFactor([Block(cats)], jnp.zeros((n, c), dtype=jnp.float32).at[:, 0].set(0.2)),
            _linear_factor(jnp.zeros(n, dtype=jnp.float32), gauss_block),
            _quadratic_factor(jnp.full(n, 0.9, dtype=jnp.float32), gauss_block),
        ],
        [],
    )
    keys = jax.random.split(jax.random.key(104), 9)
    init = [
        jax.random.bernoulli(keys[0], 0.5, (1,)).astype(jnp.bool_),
        jax.random.randint(keys[1], (1,), 0, c, dtype=jnp.uint8),
        jax.random.normal(keys[2], (1,), dtype=jnp.float32) * 0.2,
        jax.random.bernoulli(keys[3], 0.5, (1,)).astype(jnp.bool_),
        jax.random.randint(keys[4], (1,), 0, c, dtype=jnp.uint8),
        jax.random.normal(keys[5], (1,), dtype=jnp.float32) * 0.2,
        jax.random.bernoulli(keys[6], 0.5, (1,)).astype(jnp.bool_),
        jax.random.randint(keys[7], (1,), 0, c, dtype=jnp.uint8),
        jax.random.normal(keys[8], (1,), dtype=jnp.float32) * 0.2,
    ]
    return Demo(
        name="Mixed (spin+cat+gaussian)",
        program=program,
        schedule=SCHEDULE,
        init_state_free=init,
        state_clamp=[],
        nodes_to_sample=[Block(spins), Block(cats), Block(gauss)],
    )


# ── Runners ───────────────────────────────────────────────────────────────────

def _clone(state: list) -> list:
    return [jnp.array(np.asarray(s).copy()) for s in state]


def run_cpu(demo: Demo, key) -> tuple[list, float]:
    t0 = time.perf_counter()
    out = cpu_sample_states(
        key, demo.program, demo.schedule,
        _clone(demo.init_state_free), _clone(demo.state_clamp),
        demo.nodes_to_sample,
    )
    return [np.asarray(s) for s in out], time.perf_counter() - t0


def run_wormhole(demo: Demo, key, *, ttnn, device, config, signpost, profile: bool = False) -> tuple[list, float]:
    signpost(header=f"compile:{demo.name}")
    executor = tt_thrml.make_executor(ttnn, device, demo.program, config, profile=profile)

    signpost(header=f"run:{demo.name}")
    ttnn.start_tracy_zone(__file__, demo.name, 0)
    t0 = time.perf_counter()
    out = executor.sample_states(
        key, demo.schedule, demo.nodes_to_sample,
        init_state_free=_clone(demo.init_state_free),
        state_clamp=_clone(demo.state_clamp),
    )
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0
    ttnn.stop_tracy_zone(demo.name)
    signpost(header=f"done:{demo.name}")

    if profile:
        summary = executor.timing_summary()
        if summary:
            print("\n  [kernel timing]")
            for family, s in sorted(summary.items()):
                print(
                    f"    {family:<12}  n={s['n']:>4}  "
                    f"dispatch {s['dispatch_mean_ms']:>6.3f} ms/call  "
                    f"kernel {s['kernel_mean_ms']:>6.1f} ms/call  "
                    f"dispatch/total {100*s['dispatch_total_ms']/(s['dispatch_total_ms']+s['kernel_total_ms']):.1f}%"
                )

    return [np.asarray(s) for s in out], elapsed


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _spin_mean(samples: np.ndarray) -> np.ndarray:
    return np.where(samples.astype(bool), 1.0, -1.0).mean(axis=0)


def _cat_entropy(samples: np.ndarray, n_categories: int) -> float:
    probs = np.array([np.bincount(samples[:, i], minlength=n_categories) for i in range(samples.shape[1])], dtype=float)
    probs /= probs.sum(axis=1, keepdims=True) + 1e-12
    return float(-np.sum(probs * np.log(probs + 1e-12)) / samples.shape[1])


def _gauss_stats(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return samples.mean(axis=0), samples.std(axis=0)


# ── Output ────────────────────────────────────────────────────────────────────

def _print_row(label: str, value: str) -> None:
    print(f"  {label:<22} {value}")


def _report_block(s: np.ndarray) -> None:
    if s.dtype == bool or s.dtype == np.bool_:
        _print_row("spin mean", np.array2string(_spin_mean(s), precision=2, suppress_small=True))
    elif np.issubdtype(s.dtype, np.integer):
        n_cat = int(s.max()) + 1
        mode = np.array([np.bincount(s[:, i], minlength=n_cat).argmax() for i in range(s.shape[1])])
        ent = _cat_entropy(s, n_cat)
        _print_row("cat mode", np.array2string(mode))
        _print_row("cat entropy/node", f"{ent:.3f} nats")
    else:
        mean, std = _gauss_stats(s.astype(float))
        _print_row("gauss mean", np.array2string(mean, precision=3, suppress_small=True))
        _print_row("gauss std", np.array2string(std, precision=3, suppress_small=True))


def _report(demo: Demo, cpu_out: list | None, cpu_s: float | None, tt_out: list | None, tt_s: float | None) -> None:
    print(f"\n{'─'*60}")
    print(f"  {demo.name}")
    print(f"  {demo.schedule.n_samples} samples, {demo.schedule.n_warmup} warmup, {demo.schedule.steps_per_sample} steps/sample")
    print()

    for label, (out, elapsed) in [("CPU (JAX)", (cpu_out, cpu_s)), ("Wormhole", (tt_out, tt_s))]:
        if out is None:
            continue
        print(f"  [{label}]  {elapsed * 1000:.1f} ms")
        for s in out:
            _report_block(s)

    if cpu_s is not None and tt_s is not None:
        speedup = cpu_s / tt_s
        diff_ms = abs(cpu_s - tt_s) * 1000
        if speedup > 1:
            print(f"\n  speedup: {speedup:.1f}x  ({diff_ms:.0f} ms faster on Wormhole)")
        else:
            print(f"\n  CPU faster by {1/speedup:.1f}x  (Wormhole overhead: {diff_ms:.0f} ms)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cpu_only = os.environ.get("TT_DEMO_CPU_ONLY", "0") == "1"
    profile = os.environ.get("TT_DEMO_PROFILE", "0") == "1"
    device_id = int(os.environ.get("TT_DEMO_DEVICE_ID", "0"))
    artifact_root = Path(os.environ.get("TT_DEMO_ARTIFACT_ROOT", "/tmp/tt-thrml-demo"))

    demos = [_ising_chain(), _categorical_ebm(), _gaussian_chain(), _mixed_program()]
    sample_keys = jax.random.split(jax.random.key(999), len(demos))

    print("=" * 60)
    print("  THRML demo: CPU (JAX) vs Wormhole (tt_thrml)")
    print("=" * 60)

    ttnn = device = config = signpost = None

    if not cpu_only:
        import ttnn as _ttnn

        try:
            from tracy import signpost as _sp
            signpost = _sp
        except ImportError:
            def signpost(**_):
                pass

        system_desc = os.environ.get("SYSTEM_DESC_PATH")
        build_dir = os.environ.get("TTMLIR_BUILD_DIR")
        if not system_desc or not build_dir:
            print("SYSTEM_DESC_PATH and TTMLIR_BUILD_DIR required for Wormhole path.")
            print("Set TT_DEMO_CPU_ONLY=1 to run CPU baseline only.")
            sys.exit(1)

        ttnn = _ttnn
        artifact_root.mkdir(parents=True, exist_ok=True)
        config = tt_thrml.make_ttmlir_config(
            system_desc_path=system_desc,
            artifact_root=artifact_root,
            build_dir=build_dir,
        )
        device = tt_thrml.open_device(ttnn, device_id=device_id)
        print(f"  Device {device_id} open. Artifacts: {artifact_root}")

    try:
        for demo, key in zip(demos, sample_keys):
            cpu_key, tt_key = jax.random.split(key)

            # CPU warmup (JIT)
            _ = run_cpu(demo, cpu_key)
            cpu_out, cpu_s = run_cpu(demo, cpu_key)

            tt_out = tt_s = None
            if not cpu_only:
                tt_out, tt_s = run_wormhole(demo, tt_key, ttnn=ttnn, device=device, config=config, signpost=signpost, profile=profile)

            _report(demo, cpu_out, cpu_s, tt_out, tt_s)

    finally:
        if device is not None:
            ttnn.ReadDeviceProfiler(device)
            tt_thrml.close_device(ttnn, device)

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
