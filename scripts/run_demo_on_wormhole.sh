#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_demo_on_wormhole.sh

Required environment:
  TTMLIR_BUILD_DIR    Path to a TT-MLIR build with a runtime-enabled ttrt wheel
                      and matching third_party/tt-metal checkout.
  TTMLIR_SOURCE_DIR   Path to the tt-mlir source checkout (for env/activate and
                      the third_party ttnn Python packages).
  SYSTEM_DESC_PATH    Path to the target system_desc.ttsys file.

Optional environment:
  TT_THRML_CONTAINER_TOOL  Container runtime to use. Default: podman
  TT_THRML_TT_IMAGE        TT container image providing the libc/libstdc++ ABI
                           that the build was compiled against.
                           Default:
                           ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc
  TT_DEMO_DEVICE_ID        Wormhole device ID. Default: 0
  TT_DEMO_ARTIFACT_ROOT    Artifact cache root inside the container. Default: /tmp/tt-thrml-demo

Examples:
  TTMLIR_SOURCE_DIR=/path/to/tt-mlir \
  TTMLIR_BUILD_DIR=/path/to/tt-mlir/build-py310-stablehlo \
  SYSTEM_DESC_PATH=/path/to/system_desc.ttsys \
  ./scripts/run_demo_on_wormhole.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_tool="${TT_THRML_CONTAINER_TOOL:-podman}"
tt_image="${TT_THRML_TT_IMAGE:-ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc}"

: "${TTMLIR_SOURCE_DIR:?TTMLIR_SOURCE_DIR is required}"
: "${TTMLIR_BUILD_DIR:?TTMLIR_BUILD_DIR is required}"
: "${SYSTEM_DESC_PATH:?SYSTEM_DESC_PATH is required}"

source_dir="$(cd "${TTMLIR_SOURCE_DIR}" && pwd)"
build_dir="$(cd "${TTMLIR_BUILD_DIR}" && pwd)"
system_desc_path="$(cd "$(dirname "${SYSTEM_DESC_PATH}")" && pwd)/$(basename "${SYSTEM_DESC_PATH}")"
host_site_packages="$(python3 -c 'import site; print(site.getusersitepackages())')"

if [[ ! -d "${source_dir}" ]]; then
  echo "TTMLIR_SOURCE_DIR does not exist: ${source_dir}" >&2
  exit 1
fi

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

artifact_root="${TT_DEMO_ARTIFACT_ROOT:-${repo_root}/.tt-demo-artifacts}"
ttnn_jit_cache="${TT_THRML_TTNN_JIT_CACHE:-${repo_root}/.tt-demo-jit-cache}"
mkdir -p "${artifact_root}" "${ttnn_jit_cache}"

container_args=(
  "${container_tool}" run --rm
  --privileged
  --network host
  -w "${repo_root}"
  -e TTMLIR_SOURCE_DIR="${source_dir}"
  -e TTMLIR_BUILD_DIR="${build_dir}"
  -e SYSTEM_DESC_PATH="${system_desc_path}"
  -e TT_THRML_REPO_ROOT="${repo_root}"
  -e TT_THRML_TTRT_WHEEL="${ttrt_wheel}"
  -e TT_DEMO_DEVICE_ID="${TT_DEMO_DEVICE_ID:-0}"
  -e TT_DEMO_ARTIFACT_ROOT="${artifact_root}"
  -e TT_DEMO_PROFILE="${TT_DEMO_PROFILE:-0}"
  -v "${repo_root}:${repo_root}${mount_suffix}"
  -v "${source_dir}:${source_dir}${mount_suffix}"
  -v "${build_dir}:${build_dir}${mount_suffix}"
  -v "$(dirname "${system_desc_path}")":"$(dirname "${system_desc_path}")${mount_suffix}"
  -v "${ttnn_jit_cache}:/root/.cache/tt-metal-cache${mount_suffix}"
)

if [[ -n "${host_site_packages:-}" && -d "${host_site_packages}" ]]; then
  container_args+=(-v "${host_site_packages}:${host_site_packages}${mount_suffix}")
  container_args+=(-e HOST_SITE_PACKAGES="${host_site_packages}")
fi

for hugepages_path in /dev/hugepages /dev/hugepages-1G; do
  if [[ -e "${hugepages_path}" ]]; then
    container_args+=(-v "${hugepages_path}:${hugepages_path}")
  fi
done

device_id="${TT_DEMO_DEVICE_ID:-0}"
device_path="/dev/tenstorrent/${device_id}"
if [[ -e "${device_path}" ]]; then
  container_args+=(--device "${device_path}:${device_path}")
fi

container_args+=(
  "${tt_image}"
  bash -lc '
    set -euo pipefail

    build_dir_name="$(basename "${TTMLIR_BUILD_DIR}")"
    cd "${TTMLIR_SOURCE_DIR}"
    export BUILD_DIR="${build_dir_name}"
    set +u
    source env/activate 2>/dev/null || true
    set -u

    python3 -m ensurepip --upgrade >/dev/null 2>&1 || true
    python3 -m pip install --no-deps --quiet "${TT_THRML_TTRT_WHEEL}"

    export PYTHONPATH="${TT_THRML_REPO_ROOT}:${PYTHONPATH:-}"
    if [[ -n "${HOST_SITE_PACKAGES:-}" ]]; then
      export PYTHONPATH="${PYTHONPATH}:${HOST_SITE_PACKAGES}"
    fi

    cd "${TT_THRML_REPO_ROOT}"
    python3 scripts/demo.py
  '
)

exec "${container_args[@]}"
