# tt-thrml

`tt-thrml` is the Tenstorrent execution backend for upstream [`thrml`](https://github.com/extropic-ai/thrml).

Upstream `thrml` owns model authoring, blocks, samplers, and schedules. `tt-thrml` owns:
- program compilation into fused per-sampling-group TT-MLIR kernels
- bulk RNG generation (uploaded once, consumed by offset)
- device-resident global state across sweeps
- observation and sample materialization back to host

## Public Surface

The intended API is intentionally small:

- `tt_thrml.TTMLIRConfig`
- `tt_thrml.make_ttmlir_config`
- `tt_thrml.open_device`
- `tt_thrml.open_devices`
- `tt_thrml.open_mesh_device`
- `tt_thrml.close_device`
- `tt_thrml.close_mesh_device`
- `tt_thrml.close_devices`
- `tt_thrml.Executor`
- `tt_thrml.make_executor`
- `tt_thrml.MeshExecutor`
- `tt_thrml.make_mesh_executor`
- `tt_thrml.sample_states`
- `tt_thrml.sample_with_observation`
- `tt_thrml.GaussianConditional`

Everything else should be treated as compiler/runtime internals.

## Execution Model

Each THRML program is compiled once to a set of fused TT-MLIR flatbuffer kernels, one per THRML sampling group. A single sweep invokes each group kernel as:

```
(global_state, *rng_slices) -> new_global_state
```

All interaction math, Gibbs sampling, and state update happen inside the flatbuffer. Blocks in the same THRML superblock read the same pre-group state and commit together. The host never reads state back mid-sweep.

The supported parameter families are spin, categorical, and gaussian. Mixed programs built from those families are supported.

## TT-Lang Direction

The TT-Lang backend work uses a different internal state contract from the
current TT-MLIR path. Host-facing THRML state stays the same, but
device-resident state is lowered into family-specific lanes:

- spin: one signed lane per node
- categorical: one one-hot lane per category per node
- gaussian: one value lane per node

This keeps categorical source selection as direct tiled arithmetic
(`one_hot * weights`) instead of scalar category-id gather logic. The initial
layout and conversion primitives live in `tt_thrml.ttlang_backend`.

The first hardware runners are:

```bash
python scripts/run_ttlang_spin_categorical_plan.py
python scripts/run_ttlang_categorical_spin_plan.py
python scripts/run_ttlang_discrete_sweep.py
python scripts/run_ttlang_discrete_sweep.py --benchmark 50
```

They lower the shared mixed spin/categorical/gaussian smoke program into the
first TT-Lang plan shapes:

- spin target from one-hot categorical source lanes
- categorical target from signed spin source lanes
- a two-group discrete sweep that keeps the 10-lane backend state resident on
  device between group updates

The validated dispatch jobs were:

```text
j-quietbox-ttlang-spin-categorical-plan-hw-shared-program-hj4zsy
j-quietbox-ttlang-categorical-spin-plan-hw-scorebuf-hj4hzj
j-quietbox-ttlang-discrete-sweep-hw-copy3-hjmuox
j-quietbox-ttlang-discrete-sweep-bench-50-hjvh1e
j-quietbox-ttlang-discrete-sweep-bench-50-after-reboot-hkw0q6
j-quietbox-ttmlir-demo-profile-after-reboot-hkwz4n
j-quietbox-ttlang-discrete-runtime-hw-globalttl-idpbb3
j-quietbox-ttlang-discrete-runtime-bench-50-idogfw
j-quietbox-ttlang-discrete-runtime-current-state-bench-idy121
j-quietbox-ttlang-discrete-runtime-final-state-bench-ie1v9y
j-quietbox-ttlang-discrete-runtime-support-boundary-bench-iere11
j-quietbox-ttlang-discrete-runtime-randomness-bench-ieu9sz
```

The latest final-state-checked nonzero-randomness 50-sweep TT-Lang benchmark
measured 32.15 ms total, or 0.643 ms/sweep, for the current narrow
implementation. It still uses six dispatches per sweep, so this is a baseline
before fusing group copy/update work.

The current TT-MLIR/TTRT demo profile is not the same narrow workload, but it
shows the overhead we are trying to escape: the mixed demo reported mean
dispatch costs of 0.665 ms for spin, 1.251 ms for categorical, and 0.684 ms for
gaussian block calls. That makes the TT-Lang discrete runner directionally
promising even before dispatch fusion.

One detail matters for the final executor: `ttl.math.sign(0)` returns `0`,
while THRML's spin update uses a strict `>` decision whose tie result is the
negative spin. The runner encodes the strict decision as
`sign(sign(x) - 0.5)`, avoiding `where` while preserving THRML tie behavior.

Before making TT-Lang the default backend, compare it against the current
TT-MLIR/TTRT backend on the same Wormhole for spin-only, categorical-only, and
mixed workloads, measuring compile time, first-sweep latency, steady-state
milliseconds per sweep, dispatch count, transfer count, and parity against the
CPU sampler.

## Install

```bash
pip install -e ".[runtime]"
```

Available extras:
- `runtime`: installs JAX and Torch
- `jax`: installs JAX only
- `torch`: installs Torch only
- `testing`: installs the local pytest/coverage stack

TTNN, TTRT, and TT-MLIR remain environment-provided dependencies. Use a Tenstorrent container or a local Tenstorrent toolchain build for those.

## Quick Start

```python
import jax
import thrml
import ttnn

import tt_thrml

key = jax.random.key(0)
schedule = thrml.SamplingSchedule(n_warmup=32, n_samples=64, steps_per_sample=2)

config = tt_thrml.make_ttmlir_config(
    system_desc_path="/path/to/system_desc.ttsys",
    artifact_root="/tmp/tt-thrml-artifacts",
    build_dir="/path/to/tt-mlir/build",
)

device = tt_thrml.open_device(ttnn, device_id=0)
try:
    samples = tt_thrml.sample_states(
        key,
        program,
        schedule,
        init_state_free,
        state_clamp,
        [thrml.Block(nodes_to_sample)],
        ttnn=ttnn,
        device=device,
        config=config,
    )
finally:
    tt_thrml.close_device(ttnn, device)
```

## Wormhole Parity Tests

The hardware parity suite lives in [`tests/parity/test_wormhole_parity.py`](tests/parity/test_wormhole_parity.py). It requires:

```bash
export TTMLIR_BUILD_DIR=/path/to/tt-mlir/build
export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys
```

Then run with:

```bash
./scripts/run_wormhole_smoke_perf_in_tt_docker.sh -k wormhole_parity -q
```

The helper mounts the repo, TT-MLIR build, devices, and hugepages into a TT container, installs the `ttrt` wheel, and runs the suite.

## Device Ownership

- Caller-owned TTNN devices remain caller-owned.
- Executors borrow devices; they do not close them.
- The TT-MLIR bridge wraps a live TTNN device but does not take ownership.

## RNG Contract

Sampling randomness is pre-generated for the full run interval, uploaded once, and consumed by sweep offset. The executor uses a stable `(sweep, block)` index derived from the root JAX key matching upstream THRML's key derivation, so seeded runs are reproducible.

## Multi-Device

`tt-thrml` supports:
- many-job execution across multiple devices
- single-process `MeshDevice` execution with replicated state and sweep-group synchronization via `MeshExecutor`

## Internals

See [`docs/tt-thrml-internals.md`](docs/tt-thrml-internals.md) for architecture notes.
