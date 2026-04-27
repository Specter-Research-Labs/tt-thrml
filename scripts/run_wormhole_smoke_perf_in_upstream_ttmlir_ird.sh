#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_wormhole_smoke_perf_in_upstream_ttmlir_ird.sh [pytest args...]

Required environment:
  SYSTEM_DESC_PATH                Path to the target system_desc.ttsys file.

Optional environment:
  TT_THRML_CONTAINER_TOOL         Container runtime to use. Default: podman
  TT_THRML_TTMLIR_IMAGE           Upstream TT-MLIR IRD image. Default:
                                  ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-24-04:latest
  TT_THRML_UPSTREAM_TTMLIR_REF    Upstream tt-mlir git ref to build. Default: main
                                  For real Wormhole runs, prefer an explicit
                                  release tag or commit SHA over main.
  TT_THRML_UPSTREAM_TTMLIR_GIT_URL Upstream tt-mlir repo URL. Default:
                                  https://github.com/tenstorrent/tt-mlir.git
  TT_THRML_UPSTREAM_CACHE_ROOT    Host cache root for upstream checkout/builds. Default:
                                  <repo>/.tt-upstream
  TT_THRML_UPSTREAM_BUILD_DIR_NAME Build directory under the upstream checkout. Default:
                                  build-runtime-perf
  TT_THRML_BUILD_JOBS             Build parallelism inside the container. Default: nproc
  TT_THRML_USE_TRACY              Set to 1 to run pytest under `python3 -m tracy -m pytest`.
  TT_THRML_TEST_DEVICE_IDS        Passed through to the hardware test harness.
  TT_THRML_TEST_MESH_SHAPE        Passed through to the hardware test harness.
  TT_METAL_SLOW_DISPATCH_MODE     Passed through to select slow dispatch mode.
  TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN Passed through to avoid retraining flaky ETH links.
  TT_METAL_DEVICE_PROFILER        Passed through to enable device-profiler CSV dumps.
  TT_VISIBLE_DEVICES              Host-side fallback for selecting mounted TT devices.

Examples:
  SYSTEM_DESC_PATH=/path/to/system_desc.ttsys \
  TT_THRML_UPSTREAM_TTMLIR_REF=<pinned-sha-or-tag> \
  ./scripts/run_wormhole_smoke_perf_in_upstream_ttmlir_ird.sh -k spin_single_device -q

  SYSTEM_DESC_PATH=/path/to/system_desc.ttsys \
  TT_THRML_TTMLIR_IMAGE=ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-24-04:latest \
  TT_THRML_UPSTREAM_TTMLIR_REF=<pinned-sha-or-tag> \
  TT_THRML_USE_TRACY=1 \
  ./scripts/run_wormhole_smoke_perf_in_upstream_ttmlir_ird.sh -k mixed_single_device -q
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_tool="${TT_THRML_CONTAINER_TOOL:-podman}"
ttmlir_image="${TT_THRML_TTMLIR_IMAGE:-ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-24-04:latest}"
upstream_ref="${TT_THRML_UPSTREAM_TTMLIR_REF:-main}"
upstream_git_url="${TT_THRML_UPSTREAM_TTMLIR_GIT_URL:-https://github.com/tenstorrent/tt-mlir.git}"
cache_root="${TT_THRML_UPSTREAM_CACHE_ROOT:-${repo_root}/.tt-upstream}"
build_dir_name="${TT_THRML_UPSTREAM_BUILD_DIR_NAME:-build-runtime-perf}"

: "${SYSTEM_DESC_PATH:?SYSTEM_DESC_PATH is required}"

system_desc_path="$(cd "$(dirname "${SYSTEM_DESC_PATH}")" && pwd)/$(basename "${SYSTEM_DESC_PATH}")"
cache_root="$(mkdir -p "${cache_root}" && cd "${cache_root}" && pwd)"

if [[ ! -f "${system_desc_path}" ]]; then
  echo "SYSTEM_DESC_PATH does not exist: ${system_desc_path}" >&2
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
  -e SYSTEM_DESC_PATH="${system_desc_path}"
  -e TT_THRML_REPO_ROOT="${repo_root}"
  -e TT_THRML_UPSTREAM_CACHE_ROOT="${cache_root}"
  -e TT_THRML_UPSTREAM_TTMLIR_REF="${upstream_ref}"
  -e TT_THRML_UPSTREAM_TTMLIR_GIT_URL="${upstream_git_url}"
  -e TT_THRML_UPSTREAM_BUILD_DIR_NAME="${build_dir_name}"
  -e TT_THRML_USE_TRACY="${TT_THRML_USE_TRACY:-0}"
  -e TT_THRML_BUILD_JOBS="${TT_THRML_BUILD_JOBS:-}"
  -v "${repo_root}:${repo_root}${mount_suffix}"
  -v "${cache_root}:${cache_root}${mount_suffix}"
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
  "${ttmlir_image}"
  bash -lc '
    set -euo pipefail

    upstream_src="${TT_THRML_UPSTREAM_CACHE_ROOT}/tt-mlir"
    build_dir="${upstream_src}/${TT_THRML_UPSTREAM_BUILD_DIR_NAME}"

    if [[ ! -d "${upstream_src}/.git" ]]; then
      git clone "${TT_THRML_UPSTREAM_TTMLIR_GIT_URL}" "${upstream_src}"
    fi

    git -C "${upstream_src}" fetch --tags origin
    if git -C "${upstream_src}" rev-parse --verify "${TT_THRML_UPSTREAM_TTMLIR_REF}^{commit}" >/dev/null 2>&1; then
      git -C "${upstream_src}" checkout --detach "${TT_THRML_UPSTREAM_TTMLIR_REF}"
    else
      git -C "${upstream_src}" fetch origin "${TT_THRML_UPSTREAM_TTMLIR_REF}"
      git -C "${upstream_src}" checkout --detach FETCH_HEAD
    fi

    echo "Building upstream tt-mlir at:"
    git -C "${upstream_src}" show -s --format="%H%n%cs%n%s" HEAD

    cd "${upstream_src}"
    set +u
    source env/activate
    set -u

    python3 -m pip install -e "${TT_THRML_REPO_ROOT}[runtime,testing]"

    cmake -G Ninja \
      -S "${upstream_src}" \
      -B "${build_dir}" \
      -DTTMLIR_ENABLE_RUNTIME=ON \
      -DTT_RUNTIME_ENABLE_PERF_TRACE=ON \
      -DTTMLIR_ENABLE_STABLEHLO=ON \
      -DCMAKE_BUILD_TYPE=Release

    build_jobs="${TT_THRML_BUILD_JOBS:-$(nproc)}"
    cmake --build "${build_dir}" -- -j"${build_jobs}"

    tt_metal_src="${upstream_src}/third_party/tt-metal/src/tt-metal"
    export TT_METAL_HOME="${tt_metal_src}"
    export PYTHONPATH="${build_dir}/tools/ttrt/build/lib:${build_dir}/python_packages:${build_dir}/runtime/python:${tt_metal_src}/ttnn:${tt_metal_src}/tools:${PYTHONPATH:-}"
    export TTMLIR_BUILD_DIR="${build_dir}"
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

    cd "${TT_THRML_REPO_ROOT}"
    if [[ "${TT_THRML_USE_TRACY:-0}" == "1" ]]; then
      python3 -m tracy -m pytest tests/parity/test_wormhole_parity.py "$@"
    else
      python3 -m pytest tests/parity/test_wormhole_parity.py "$@"
    fi
  ' bash "$@"
)

exec "${container_args[@]}"
