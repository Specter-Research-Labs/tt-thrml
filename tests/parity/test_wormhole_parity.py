from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = [pytest.mark.hardware, pytest.mark.slow]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _extract_json_payload(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        payload = line.strip()
        if not payload:
            continue
        if not (payload.startswith("{") and payload.endswith("}")):
            continue
        return json.loads(payload)
    raise AssertionError(
        "parity runner did not emit a JSON payload.\n"
        f"stdout:\n{stdout}"
    )


@pytest.fixture(scope="module")
def wormhole_parity_results():
    if not os.environ.get("SYSTEM_DESC_PATH"):
        pytest.skip("SYSTEM_DESC_PATH is required for Wormhole parity tests.")
    if not os.environ.get("TTMLIR_BUILD_DIR"):
        pytest.skip("TTMLIR_BUILD_DIR is required for Wormhole parity tests.")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "tests.parity._parity_runner"],
            capture_output=True,
            cwd=REPO_ROOT,
            text=True,
            check=False,
            timeout=1800,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "parity runner timed out.\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc
    if completed.returncode != 0:
        raise AssertionError(
            "parity runner failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return _extract_json_payload(completed.stdout)


def test_spin_sample_distribution_matches_upstream_thrml_on_wormhole(
    wormhole_parity_results,
):
    diffs = wormhole_parity_results["spin"]["diffs"]
    assert diffs["state_hist_linf"] <= 0.05
    assert diffs["mean_signed_linf"] <= 0.02


def test_categorical_sample_distribution_matches_upstream_thrml_on_wormhole(
    wormhole_parity_results,
):
    diffs = wormhole_parity_results["categorical"]["diffs"]
    assert diffs["state_hist_linf"] <= 0.05
    assert diffs["node_hist_linf"] <= 0.03


def test_gaussian_sample_moments_match_upstream_thrml_on_wormhole(
    wormhole_parity_results,
):
    diffs = wormhole_parity_results["gaussian"]["diffs"]
    assert diffs["mean_linf"] <= 5.0e-3
    assert diffs["cov_linf"] <= 1.0e-2


def test_mixed_program_statistics_match_upstream_thrml_on_wormhole(
    wormhole_parity_results,
):
    diffs = wormhole_parity_results["mixed"]["diffs"]
    assert diffs["discrete_hist_linf"] <= 0.05
    assert diffs["gaussian_mean_linf"] <= 5.0e-3
    assert diffs["gaussian_cov_linf"] <= 1.0e-2


def test_observer_and_clamped_state_moments_match_upstream_thrml_on_wormhole(
    wormhole_parity_results,
):
    diffs = wormhole_parity_results["observation_clamp"]["diffs"]
    assert diffs["first_moment_linf"] <= 0.02
    assert diffs["second_moment_linf"] <= 0.02
