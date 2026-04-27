#!/usr/bin/env python3
"""Probe whether tt-mlir can lower stablehlo.gather through to TTNN flatbuffer.

Escalates in three stages:
  stage 1: pure gather (global_state[indices])
  stage 2: gather -> multiply -> reduce (gamma computation)
  stage 3: stage 2 + jnp.where threshold (full fused sweep kernel)

For each stage that compiles, also executes on device and compares to a numpy
reference.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent))
import tt_thrml
from tt_thrml.executor import _resolve_runtime_bridge


def _lower_to_stablehlo(fn: Callable, *inputs) -> str:
    return jax.jit(fn).lower(*inputs).as_text(dialect="stablehlo")


def _run_pipeline(
    stablehlo_text: str, artifact_dir: Path, system_desc_path: str, ttmlir_opt: str, ttmlir_translate: str
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stablehlo_path = artifact_dir / "k.stablehlo.mlir"
    ttir_path = artifact_dir / "k.ttir.mlir"
    ttnn_path = artifact_dir / "k.ttnn.mlir"
    flatbuffer_path = artifact_dir / "k.ttnn"
    stablehlo_path.write_text(stablehlo_text)

    subprocess.run(
        [ttmlir_opt, "--stablehlo-to-ttir-pipeline", str(stablehlo_path), "-o", str(ttir_path)],
        check=True,
        text=True,
    )
    subprocess.run(
        [
            ttmlir_opt,
            f"--ttir-to-ttnn-backend-pipeline=enable-cpu-hoisted-const-eval=false system-desc-path={system_desc_path}",
            str(ttir_path),
            "-o",
            str(ttnn_path),
        ],
        check=True,
        text=True,
    )
    subprocess.run(
        [ttmlir_translate, "--ttnn-to-flatbuffer", str(ttnn_path), "-o", str(flatbuffer_path)],
        check=True,
        text=True,
    )
    return flatbuffer_path


def _run_on_device(flatbuffer_path: Path, device, input_tensors: list) -> object:
    import ttrt.runtime as tt_runtime
    from ttrt.common.util import Binary, FileManager, Logger

    runtime_utils = _resolve_runtime_bridge(tt_runtime)

    logger = Logger()
    binary = Binary(logger, FileManager(logger), str(flatbuffer_path))
    tt_runtime.set_compatible_device_runtime(binary.fbb)
    runtime_device = runtime_utils.create_runtime_device_from_ttnn(device)

    runtime_inputs = []
    for i, t in enumerate(input_tensors):
        rt = runtime_utils.create_runtime_tensor_from_ttnn(t, True)
        layout = tt_runtime.get_layout(binary.fbb, 0, i)
        runtime_inputs.append(tt_runtime.to_layout(rt, runtime_device, layout, True))

    try:
        outs = tt_runtime.submit(runtime_device, binary.fbb, 0, runtime_inputs)
        ttnn_outs = [runtime_utils.get_ttnn_tensor_from_runtime_tensor(o) for o in outs]
    finally:
        for rt in runtime_inputs:
            tt_runtime.deallocate_tensor(rt, force=True)
    return ttnn_outs


def _upload(device, arr: np.ndarray, layout=None) -> object:
    if layout is None:
        layout = ttnn.TILE_LAYOUT
    tensor = torch.from_numpy(arr.astype(np.float32).copy()).contiguous()
    return ttnn.from_torch(tensor, dtype=ttnn.float32, layout=layout, device=device)


def _download(t) -> np.ndarray:
    return ttnn.to_torch(t).cpu().numpy()


def stage1_pure_gather(
    artifact_root: Path, system_desc_path: str, ttmlir_opt: str, ttmlir_translate: str, device
) -> None:
    print("\n=== stage 1: pure gather ===")

    global_state_np = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    indices_np = np.array([[3, 0], [4, 3], [4, 0]], dtype=np.int32)

    indices_const = jnp.asarray(indices_np)

    def fn(global_state):
        return global_state[indices_const]

    inputs = [jax.ShapeDtypeStruct(global_state_np.shape, jnp.float32)]
    stablehlo = _lower_to_stablehlo(fn, *inputs)
    print("StableHLO (first 400 chars):")
    print(stablehlo[:400])

    fb = _run_pipeline(stablehlo, artifact_root / "stage1", system_desc_path, ttmlir_opt, ttmlir_translate)
    print(f"compiled -> {fb}")

    gs_tensor = _upload(device, global_state_np)
    outs = _run_on_device(fb, device, [gs_tensor])
    result = _download(outs[0])
    expected = global_state_np[indices_np]
    print(f"result: {result}")
    print(f"expected: {expected}")
    if np.allclose(result.reshape(expected.shape), expected):
        print("STAGE 1 OK")
    else:
        print("STAGE 1 MISMATCH")


def stage2_gather_arith(
    artifact_root: Path, system_desc_path: str, ttmlir_opt: str, ttmlir_translate: str, device
) -> None:
    print("\n=== stage 2: gather -> multiply -> reduce (gamma) ===")

    # 5-node Ising chain: block 0 = nodes 0,2,4 (3 targets), block 1 = nodes 1,3
    global_state_np = np.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    gather_indices_np = np.array([[3, 0], [4, 3], [4, 0]], dtype=np.int32)
    weights_np = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    mask_np = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    gi = jnp.asarray(gather_indices_np)
    w = jnp.asarray(weights_np)
    m = jnp.asarray(mask_np)

    def fn(global_state):
        gathered = global_state[gi]
        scale = w * m * gathered
        gamma = jnp.sum(scale, axis=-1, keepdims=True)
        return gamma

    inputs = [jax.ShapeDtypeStruct(global_state_np.shape, jnp.float32)]
    stablehlo = _lower_to_stablehlo(fn, *inputs)
    print("StableHLO (first 600 chars):")
    print(stablehlo[:600])

    fb = _run_pipeline(stablehlo, artifact_root / "stage2", system_desc_path, ttmlir_opt, ttmlir_translate)
    print(f"compiled -> {fb}")

    gs_tensor = _upload(device, global_state_np)
    outs = _run_on_device(fb, device, [gs_tensor])
    result = _download(outs[0]).squeeze()
    expected = np.sum(weights_np * mask_np * global_state_np[gather_indices_np], axis=-1)
    print(f"result: {result}")
    print(f"expected: {expected}")
    if np.allclose(result, expected, atol=1e-4):
        print("STAGE 2 OK")
    else:
        print("STAGE 2 MISMATCH")


def stage3_full_sweep(
    artifact_root: Path, system_desc_path: str, ttmlir_opt: str, ttmlir_translate: str, device
) -> None:
    print("\n=== stage 3: full fused sweep (gather + gamma + where) ===")

    global_state_np = np.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    gather_indices_np = np.array([[3, 0], [4, 3], [4, 0]], dtype=np.int32)
    weights_np = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    mask_np = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    rng_slice_np = np.array([[0.1], [-0.2], [0.3]], dtype=np.float32)

    gi = jnp.asarray(gather_indices_np)
    w = jnp.asarray(weights_np)
    m = jnp.asarray(mask_np)

    def fn(global_state, rng_slice):
        gathered = global_state[gi]
        gamma = jnp.sum(w * m * gathered, axis=-1, keepdims=True)
        return jnp.where(gamma > rng_slice, 1.0, -1.0)

    inputs = [
        jax.ShapeDtypeStruct(global_state_np.shape, jnp.float32),
        jax.ShapeDtypeStruct(rng_slice_np.shape, jnp.float32),
    ]
    stablehlo = _lower_to_stablehlo(fn, *inputs)
    print("StableHLO (first 800 chars):")
    print(stablehlo[:800])

    fb = _run_pipeline(stablehlo, artifact_root / "stage3", system_desc_path, ttmlir_opt, ttmlir_translate)
    print(f"compiled -> {fb}")

    gs_tensor = _upload(device, global_state_np)
    rng_tensor = _upload(device, rng_slice_np)
    outs = _run_on_device(fb, device, [gs_tensor, rng_tensor])
    result = _download(outs[0]).squeeze()

    gamma_ref = np.sum(weights_np * mask_np * global_state_np[gather_indices_np], axis=-1, keepdims=True)
    expected = np.where(gamma_ref > rng_slice_np, 1.0, -1.0).squeeze()
    print(f"gamma_ref: {gamma_ref.squeeze()}")
    print(f"result: {result}")
    print(f"expected: {expected}")
    if np.allclose(result, expected, atol=1e-4):
        print("STAGE 3 OK")
    else:
        print("STAGE 3 MISMATCH")


def main() -> int:
    system_desc_path = os.environ.get("SYSTEM_DESC_PATH")
    build_dir = os.environ.get("TTMLIR_BUILD_DIR")
    if not system_desc_path or not build_dir:
        print("ERROR: SYSTEM_DESC_PATH and TTMLIR_BUILD_DIR must be set")
        return 1

    ttmlir_opt = str(Path(build_dir) / "bin" / "ttmlir-opt")
    ttmlir_translate = str(Path(build_dir) / "bin" / "ttmlir-translate")
    artifact_root = Path(tempfile.mkdtemp(prefix="probe-gather-"))
    print(f"artifact_root: {artifact_root}")

    device = tt_thrml.open_device(ttnn, device_id=0)

    try:
        for stage_fn in (stage1_pure_gather, stage2_gather_arith, stage3_full_sweep):
            try:
                stage_fn(artifact_root, system_desc_path, ttmlir_opt, ttmlir_translate, device)
            except subprocess.CalledProcessError as e:
                print(f"COMPILE FAILED in {stage_fn.__name__}: exit {e.returncode}")
                print(f"  cmd: {e.cmd}")
                return 1
            except Exception:
                print(f"RUN FAILED in {stage_fn.__name__}:")
                traceback.print_exc()
                return 1
    finally:
        tt_thrml.close_device(ttnn, device)

    print("\nALL STAGES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
