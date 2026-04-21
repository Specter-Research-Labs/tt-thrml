# tt-thrml

`tt-thrml` is the Tenstorrent execution backend for upstream [`thrml`](https://github.com/extropic-ai/thrml).

Upstream `thrml` owns model authoring, blocks, samplers, and schedules. `tt-thrml` owns:
- program compilation into TT-shaped runtime metadata
- device-resident state/runtime orchestration
- TT-MLIR parameter-kernel execution
- observation and sample materialization back to host

## Public Surface

The intended API is intentionally small:

- `tt_thrml.BackendBinding`
- `tt_thrml.ExecutionOptions`
- `tt_thrml.ParameterFamily`
- `tt_thrml.ParameterKernelBackend`
- `tt_thrml.TTMLIRConfig`
- `tt_thrml.make_backend_binding`
- `tt_thrml.make_ttmlir_backend_binding`
- `tt_thrml.make_ttmlir_config`
- `tt_thrml.make_ttmlir_parameter_kernel_backends`
- `tt_thrml.make_ttmlir_parameter_kernel_ops`
- `tt_thrml.open_device`
- `tt_thrml.open_devices`
- `tt_thrml.open_mesh_device`
- `tt_thrml.close_devices`
- `tt_thrml.close_mesh_device`
- `tt_thrml.sample_states`
- `tt_thrml.sample_states_many`
- `tt_thrml.sample_with_observation`
- `tt_thrml.sample_with_observation_many`

Everything else should be treated as compiler/runtime internals.

## Execution Model

`tt-thrml` compiles a THRML program into TT runtime metadata once, keeps canonical state on device, and executes the sweep with:
- TTNN for state/orchestration/runtime plumbing
- TT-MLIR for parameter-family math kernels

The supported parameter families are:
- spin
- categorical
- gaussian

Mixed programs built from those families are supported too.

The training / moment-estimation helpers from upstream notebooks are out of scope for this repo. Sampling and observation flows are the supported surface.

## Install

```bash
pip install -e ".[runtime]"
```

Available extras:
- `runtime`: installs JAX and Torch for the public API surface
- `jax`: installs JAX only
- `torch`: installs Torch only
- `examples`: installs notebook/example-only Python dependencies
- `testing`: installs the local pytest/coverage stack

TTNN, TTRT, and TT-MLIR remain environment-provided dependencies. Use a Tenstorrent container or a local Tenstorrent toolchain build for those pieces.

## Quick Start

```python
import jax
import thrml
import ttnn

import tt_thrml

key = jax.random.key(0)
schedule = thrml.SamplingSchedule(n_warmup=32, n_samples=64, steps_per_sample=2)

device = tt_thrml.open_device(ttnn, device_id=0)
backend = tt_thrml.make_ttmlir_backend_binding(
    ttnn,
    device,
    system_desc_path="/path/to/system_desc.ttsys",
    artifact_root="/tmp/tt-thrml-artifacts",
    build_dir="/path/to/tt-mlir/build-py310-stablehlo",
)

try:
    samples = tt_thrml.sample_states(
        key,
        program,
        schedule,
        init_state_free,
        state_clamp,
        [thrml.Block(nodes_to_sample)],
        backend=backend,
    )
finally:
    tt_thrml.close_devices(ttnn, (device,))
```

## Fresh QuietBox

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[runtime,examples]"

export TTMLIR_BUILD_DIR=/path/to/tt-mlir/build-py310-stablehlo
export SYSTEM_DESC_PATH=/path/to/tt-mlir/ttrt-artifacts/system_desc.ttsys

python examples/tt_ising_chain_demo.py \
  --parameter-kernels ttmlir \
  --device-id 0 \
  --repeat 1 \
  --n-warmup 8 \
  --n-samples 16 \
  --steps-per-sample 2
```

`tt-thrml` expects the TT runtime and TT-MLIR toolchain to be provided by the environment, typically through a Tenstorrent container/toolchain or a local `tt-mlir` build.

## Device Ownership

- Caller-owned TTNN devices remain caller-owned.
- Caller-owned `MeshDevice`s remain caller-owned.
- Executors borrow devices; they do not close them.
- The direct TT-MLIR bridge may temporarily wrap a live TTNN device, but it does not take ownership of it.
- `tt-thrml` only closes TT runtime sessions that it opened itself.

## RNG Contract

Sampling randomness is prepared per block for the whole run interval, uploaded once, and consumed by iteration offset inside the sweep. The executor uses a stable `(iteration, block)` key mapping derived from the root JAX key, so seeded runs remain reproducible while avoiding tiny per-step random uploads in the hot loop.

## Multi-Device

`tt-thrml` supports:
- many-job execution across multiple devices
- single-process `MeshDevice` execution with replicated state and sweep-group synchronization

It does not implement a sharded multi-Wormhole single-program sweep engine.

## Examples

- [`examples/tt_ising_chain_demo.py`](examples/tt_ising_chain_demo.py)
- [`examples/tt_categorical_ebm_demo.py`](examples/tt_categorical_ebm_demo.py)
- [`examples/tt_mixed_discrete_ebm_demo.py`](examples/tt_mixed_discrete_ebm_demo.py)
- [`examples/tt_gaussian_chain_demo.py`](examples/tt_gaussian_chain_demo.py)
- [`examples/tt_probabilistic_computing_demo.py`](examples/tt_probabilistic_computing_demo.py)
- [`examples/tt_spin_models_sampling_demo.py`](examples/tt_spin_models_sampling_demo.py)
- [`examples/tt_all_of_thrml_demo.py`](examples/tt_all_of_thrml_demo.py)
- [`examples/tt_run_existing_program.py`](examples/tt_run_existing_program.py)

## Internals

See [`docs/tt-thrml-internals.md`](docs/tt-thrml-internals.md) for a compact architecture note and execution diagrams.
