# tt-thrml

`tt-thrml` runs upstream [`thrml`](https://github.com/extropic-ai/thrml) programs on Tenstorrent hardware.

The split is simple:

- `thrml` defines the probabilistic program, blocks, factors, samplers, and schedule
- `tt-thrml` executes that program on TT hardware
- TT-MLIR is the main parameter-math path
- TTNN handles device/runtime state, sampling orchestration, and writeback

## How It Works

At a high level:

1. upstream `thrml` builds a program and a sampling schedule
2. `tt-thrml` compiles that into TT execution metadata
3. canonical state lives on device
4. for each sweep group, `tt-thrml` gathers the needed state, computes parameters, samples new block values, and writes them back
5. parameter math goes through TT-MLIR; runtime orchestration stays in `tt-thrml`

A slightly more detailed internal note is in [docs/tt-thrml-internals.md](docs/tt-thrml-internals.md).

## Supported Today

The supported surface today is the THRML sampling path:

- spin / Ising programs
- categorical discrete EBM programs
- gaussian programs
- mixed sampling programs built from those families
- observation flows

The main thing that is still out of scope is the upstream training / moment-estimation side.

## Quick Start

Assume `program`, `init_state_free`, `state_clamp`, and `nodes_to_sample` come from normal
upstream `thrml` authoring code.

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

On a fresh QuietBox (and assuming the tenstorrent toolchain is installed):

1. create a Python environment
2. `pip install -e .`
3. if you want the notebook-style example ports too, use `pip install -e ".[examples]"`
4. make sure the environment already has `ttrt.runtime` and a TT-MLIR toolchain
5. set `TTMLIR_BUILD_DIR` and `SYSTEM_DESC_PATH`
6. run one of the examples

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[examples]"

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

If you use Tenstorrent's official container, install `tt-thrml` inside that container and point it
at the matching TT-MLIR toolchain the same way.

## Examples

The repo includes both small TT demos and near-direct ports of the upstream THRML examples:

- [examples/tt_ising_chain_demo.py](examples/tt_ising_chain_demo.py)
- [examples/tt_categorical_ebm_demo.py](examples/tt_categorical_ebm_demo.py)
- [examples/tt_mixed_discrete_ebm_demo.py](examples/tt_mixed_discrete_ebm_demo.py)
- [examples/tt_gaussian_chain_demo.py](examples/tt_gaussian_chain_demo.py)
- [examples/tt_probabilistic_computing_demo.py](examples/tt_probabilistic_computing_demo.py)
- [examples/tt_spin_models_sampling_demo.py](examples/tt_spin_models_sampling_demo.py)
- [examples/tt_all_of_thrml_demo.py](examples/tt_all_of_thrml_demo.py)

## Multi-Device

`tt-thrml` has a multi-device path through TT `MeshDevice`.

Current scope:

- single-process mesh execution
- explicit TT mesh placement/composition
