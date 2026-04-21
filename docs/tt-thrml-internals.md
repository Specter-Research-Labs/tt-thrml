# TT-THRML Internals

`tt-thrml` has one simple boundary:

- upstream `thrml` owns program authoring
- `tt-thrml` owns Tenstorrent execution

The executor keeps canonical block state on device, compiles TT-shaped metadata once, and routes parameter-family math through TT-MLIR-backed kernels.

## Architecture

```mermaid
flowchart TD
    A["THRML program + schedule"] --> B["tt_thrml.api"]
    B --> C["BackendBinding"]
    C --> D["backend_executor"]
    D --> E["CompiledProgram"]
    D --> F["TTProgramExecutor / TTMeshProgramExecutor"]
    E --> F
    F --> G["state_runtime"]
    F --> H["observation_runtime"]
    F --> I["family_handlers"]
    I --> J["TT-MLIR parameter kernels"]
    I --> K["TTNN runtime primitives"]
    J --> L["StableHLO -> TTIR -> TTNN -> TTRT"]
```

## Sweep Flow

```mermaid
flowchart TD
    A["run_sweep(...)"] --> B["prepare block random buffers"]
    B --> C["for sampling group"]
    C --> D["for block"]
    D --> E["gather interaction sources from device state"]
    E --> F["compute_block_parameters(...)"]
    F --> G["sample block"]
    G --> H["stage pending block update"]
    H --> I["write group updates to canonical device state"]
    I --> J{"more groups?"}
    J -->|yes| C
    J -->|no| K["observe / materialize outputs"]
```

## Parameter-Kernel Boundary

The parameter-kernel layer is the main TT-MLIR boundary.

- The compiler emits per-block runtime metadata and physical tensor specs.
- The executor builds one block payload from grouped interactions.
- The family handler launches one block parameter op.
- The TT-MLIR bridge uses cached metadata/signatures and direct TTNN runtime bridging as the default contract.

The executor still owns schedule iteration, state writes, and observation.

## Device Ownership

- TTNN devices passed in by the caller stay caller-owned.
- `MeshDevice`s passed in by the caller stay caller-owned.
- Executors borrow those devices.
- TT-MLIR runtime sessions opened internally are owned and closed by `tt-thrml`.

## RNG Contract

Sampling randomness is prepared per block across the requested iteration interval, uploaded once, and consumed by iteration offset. The root JAX key maps deterministically to `(iteration, block)` sample keys, so runs remain reproducible without rebuilding tiny random tensors inside the sweep hot path.
