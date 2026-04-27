#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_wormhole_smoke_perf_in_tt_docker.sh [pytest args...]

Required environment:
  TTMLIR_BUILD_DIR    Path to a TT-MLIR build with a runtime-enabled ttrt wheel.
  SYSTEM_DESC_PATH    Path to the target system_desc.ttsys file.

Optional environment:
  TT_THRML_CONTAINER_TOOL  Container runtime to use. Default: podman
  TT_THRML_TT_IMAGE        TT container image that provides a compatible ttnn.
                           Default:
                           ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc
  TT_THRML_TTRT_WHEEL      Explicit ttrt wheel path. Defaults to the newest wheel
                           under $TTMLIR_BUILD_DIR/tools/ttrt/build/.
  TT_THRML_USE_TRACY       Set to 1 to run pytest under `python3 -m tracy -m pytest`.
  TT_THRML_TEST_DEVICE_IDS Passed through to the hardware test harness.
  TT_THRML_TEST_MESH_SHAPE Passed through to the hardware test harness.
  TT_METAL_SLOW_DISPATCH_MODE Passed through to select slow dispatch mode.
  TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN Passed through to avoid retraining flaky ETH links.
  TT_METAL_DEVICE_PROFILER Passed through to enable device-profiler CSV dumps.
  TT_VISIBLE_DEVICES       Host-side fallback for selecting mounted TT devices.

Examples:
  TTMLIR_BUILD_DIR=/path/to/tt-mlir/build-py310-stablehlo \
  SYSTEM_DESC_PATH=/path/to/system_desc.ttsys \
  ./scripts/run_wormhole_smoke_perf_in_tt_docker.sh -k spin_single_device -q

  TT_THRML_USE_TRACY=1 \
  TTMLIR_BUILD_DIR=/path/to/tt-mlir/build-py310-stablehlo \
  SYSTEM_DESC_PATH=/path/to/system_desc.ttsys \
  ./scripts/run_wormhole_smoke_perf_in_tt_docker.sh -k mixed_single_device -q
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_tool="${TT_THRML_CONTAINER_TOOL:-podman}"
tt_image="${TT_THRML_TT_IMAGE:-ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc}"

: "${TTMLIR_BUILD_DIR:?TTMLIR_BUILD_DIR is required}"
: "${SYSTEM_DESC_PATH:?SYSTEM_DESC_PATH is required}"

build_dir="$(cd "${TTMLIR_BUILD_DIR}" && pwd)"
system_desc_path="$(cd "$(dirname "${SYSTEM_DESC_PATH}")" && pwd)/$(basename "${SYSTEM_DESC_PATH}")"

if [[ ! -d "${build_dir}" ]]; then
  echo "TTMLIR_BUILD_DIR does not exist: ${build_dir}" >&2
  exit 1
fi

if [[ ! -f "${system_desc_path}" ]]; then
  echo "SYSTEM_DESC_PATH does not exist: ${system_desc_path}" >&2
  exit 1
fi

ttrt_wheel="${TT_THRML_TTRT_WHEEL:-}"
if [[ -z "${ttrt_wheel}" ]]; then
  shopt -s nullglob
  wheels=("${build_dir}"/tools/ttrt/build/ttrt-*.whl)
  shopt -u nullglob
  if [[ "${#wheels[@]}" -eq 0 ]]; then
    echo "No ttrt wheel found under ${build_dir}/tools/ttrt/build/" >&2
    exit 1
  fi
  ttrt_wheel="${wheels[-1]}"
fi
ttrt_wheel="$(cd "$(dirname "${ttrt_wheel}")" && pwd)/$(basename "${ttrt_wheel}")"

if [[ ! -f "${ttrt_wheel}" ]]; then
  echo "TTRT wheel does not exist: ${ttrt_wheel}" >&2
  exit 1
fi

mount_suffix=""
if [[ "$(basename "${container_tool}")" == "podman" ]]; then
  mount_suffix=":Z"
fi

container_args=(
  "${container_tool}" run --rm
  --network host
  --privileged
  -w "${repo_root}"
  -e TTMLIR_BUILD_DIR="${build_dir}"
  -e SYSTEM_DESC_PATH="${system_desc_path}"
  -e TT_THRML_REPO_ROOT="${repo_root}"
  -e TT_THRML_TTRT_WHEEL="${ttrt_wheel}"
  -e TT_THRML_USE_TRACY="${TT_THRML_USE_TRACY:-0}"
  -v "${repo_root}:${repo_root}${mount_suffix}"
  -v "${build_dir}:${build_dir}${mount_suffix}"
  -v "$(dirname "${system_desc_path}")":"$(dirname "${system_desc_path}")${mount_suffix}"
)

for maybe_env in TT_THRML_TEST_DEVICE_IDS TT_THRML_TEST_MESH_SHAPE TT_METAL_SLOW_DISPATCH_MODE TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN TT_METAL_DEVICE_PROFILER; do
  if [[ -n "${!maybe_env:-}" ]]; then
    container_args+=(-e "${maybe_env}=${!maybe_env}")
  fi
done

for hugepages_path in /dev/hugepages /dev/hugepages-1G; do
  if [[ -e "${hugepages_path}" ]]; then
    container_args+=(-v "${hugepages_path}:${hugepages_path}")
  fi
done

device_ids_csv="${TT_THRML_TEST_DEVICE_IDS:-${TT_VISIBLE_DEVICES:-0}}"
IFS=',' read -r -a device_ids <<< "${device_ids_csv}"
for device_id in "${device_ids[@]}"; do
  trimmed_id="${device_id//[[:space:]]/}"
  [[ -z "${trimmed_id}" ]] && continue
  device_path="/dev/tenstorrent/${trimmed_id}"
  if [[ -e "${device_path}" ]]; then
    container_args+=(--device "${device_path}:${device_path}")
  fi
done

container_args+=(
  "${tt_image}"
  bash -lc '
    set -euo pipefail
    python3 -m ensurepip --upgrade >/dev/null 2>&1 || true
    python3 -m pip install "${TT_THRML_REPO_ROOT}[runtime,testing]"
    python3 -m pip install --no-deps "${TT_THRML_TTRT_WHEEL}"
    cd "${TT_THRML_REPO_ROOT}"
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    if [[ "${TT_THRML_USE_TRACY:-0}" == "1" ]]; then
      python3 -m tracy -m pytest tests/parity/test_wormhole_parity.py "$@"
    else
      python3 -m pytest tests/parity/test_wormhole_parity.py "$@"
    fi
  ' bash "$@"
)

exec "${container_args[@]}"
