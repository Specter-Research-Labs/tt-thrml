# tt-thrml

`tt-thrml` is the Tenstorrent TT-Lang execution layer for upstream
[`thrml`](https://github.com/extropic-ai/thrml).

Upstream `thrml` owns model authoring, blocks, samplers, and schedules.
`tt-thrml` owns the device-resident TT-Lang execution path for supported THRML
program shapes.

## Public Surface

The intended API is intentionally small:

- `tt_thrml.open_device`
- `tt_thrml.open_devices`
- `tt_thrml.open_mesh_device`
- `tt_thrml.close_device`
- `tt_thrml.close_mesh_device`
- `tt_thrml.close_devices`
- `tt_thrml.make_executor`

Everything else is compiler/runtime internals.

## Execution Model

`make_executor` builds the primary TT-Lang executor. Unsupported program shapes
fail clearly instead of switching to another backend.

The current hardware-proven path supports the mixed discrete smoke shape:

- spin target from one-hot categorical source lanes
- categorical target from signed spin source lanes
- two or more independent Gibbs sampling groups
- device-resident 10-lane TT-Lang state across sweeps
- one TT-Lang dispatch per sweep for regular independent groups

Host-facing THRML state remains booleans, category ids, and floats. Device state
is lowered into TT-Lang lanes:

- spin: one signed lane per node
- categorical: one one-hot lane per category per node
- gaussian: one value lane per node, preserved until the TT-Lang gaussian update
  kernel lands

## Quick Start

```python
import jax
import ttnn

import tt_thrml
from tt_thrml.example_programs import make_mixed_spin_categorical_gaussian_program

program = make_mixed_spin_categorical_gaussian_program()
initial_state = [[True], [2], [0.25], [False], [0], [-0.5]]

device = tt_thrml.open_device(ttnn, device_id=0)
try:
    executor = tt_thrml.make_executor(ttnn, device, program)
    executor.load_state(initial_state)
    executor.set_sweep_randomness_window_from_key(jax.random.PRNGKey(0), 50)
    executor.run_sweeps(50)
    free_state, clamped_state = executor.read_state_lists()
finally:
    tt_thrml.close_device(ttnn, device)
```

## Runners

```bash
python scripts/run_ttlang_spin_categorical_plan.py
python scripts/run_ttlang_categorical_spin_plan.py
python scripts/run_ttlang_discrete_sweep.py
python scripts/run_ttlang_discrete_sweep.py --benchmark 50
python scripts/run_ttlang_discrete_sweep.py --pairs 3 --benchmark 50
python scripts/run_ttlang_discrete_sweep.py --warmup 10 --benchmark 100 --json
```

For QuietBox validation, run inside the `tt-lang-codex` TT-Lang container or
through `dispatch` with a Wormhole device allocation. The container must expose
the TT device, hugepages, 1G hugepage sysfs, host networking for dependency
installation, and `nofile`/`nproc` limits no higher than QuietBox's SSH hard
limit so `podman exec` works from both SSH and dispatch sessions:

```bash
podman run -d --privileged --network host --name tt-lang-codex \
  --ulimit nofile=524288:524288 \
  --ulimit nproc=524288:524288 \
  --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:latest sleep infinity
```

## Latest Hardware Check

Current cleanup HEAD passed on QuietBox for both the original two-group shape
and a three-group generalized shape. Both shapes now fuse regular independent
groups into one dispatch per sweep:

```text
default mixed discrete shape
0.274 ms/sweep, 1 dispatch/sweep

three independent group shape
0.422 ms/sweep, 1 dispatch/sweep
```

The prior two-dispatch grouped runtime measured about 0.390 ms/sweep on the
same QuietBox benchmark shape. The earlier 6-dispatch runtime measured
0.699 ms/sweep.

## TT-Lang Reproducer

[`reproducers/ttlang_simultaneous_carry_hang.py`](reproducers/ttlang_simultaneous_carry_hang.py)
captures a minimal simulator-clean, hardware-hanging simultaneous carried-CB
pattern.
[`reproducers/ttlang_coupled_recurrent_hang.py`](reproducers/ttlang_coupled_recurrent_hang.py)
captures the THRML-shaped blocker: a sequential carried spin/category recurrent
window that still hangs on hardware. Until that upstream issue is resolved, the
production runtime keeps the hardware-clean fused one-sweep path.

## Randomness

The runtime accepts explicit per-block random inputs for deterministic tests.
For chain runs, prefer a device-resident randomness window derived from a JAX
key using THRML's own schedule:

```python
runtime.set_sweep_randomness_window_from_key(jax.random.PRNGKey(0), n_sweeps)
```

This mirrors THRML's per-sweep block-key split and sampler-key split. Spin
blocks receive Bernoulli logit thresholds; categorical blocks receive Gumbel-max
perturbations.

## Install

```bash
pip install -e ".[runtime]"
```

Available extras:

- `runtime`: installs JAX and Torch
- `jax`: installs JAX only
- `torch`: installs Torch only
- `testing`: installs the local pytest/coverage stack
- `development`: installs formatters and type checking tools

TTNN and TT-Lang remain environment-provided dependencies.

## Device Ownership

- Caller-owned TTNN devices remain caller-owned.
- Executors borrow devices; they do not close them.

## Internals

See [`docs/tt-thrml-internals.md`](docs/tt-thrml-internals.md) for architecture
notes.
