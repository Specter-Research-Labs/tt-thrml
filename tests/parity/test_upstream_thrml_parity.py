from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.slow
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def parity_results():
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "tests.parity._parity_runner"],
            capture_output=True,
            cwd=REPO_ROOT,
            text=True,
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "Parity runner timed out.\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc
    if completed.returncode != 0:
        raise AssertionError(
            "Parity runner failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def test_spin_sample_distribution_matches_upstream_thrml(parity_results):
    diffs = parity_results["spin"]["diffs"]
    assert diffs["state_hist_linf"] <= 0.05
    assert diffs["mean_signed_linf"] <= 0.02


def test_categorical_sample_distribution_matches_upstream_thrml(parity_results):
    diffs = parity_results["categorical"]["diffs"]
    assert diffs["state_hist_linf"] <= 0.05
    assert diffs["node_hist_linf"] <= 0.03


def test_gaussian_sample_moments_match_upstream_thrml(parity_results):
    diffs = parity_results["gaussian"]["diffs"]
    assert diffs["mean_linf"] <= 1.0e-3
    assert diffs["cov_linf"] <= 2.0e-3


def test_mixed_program_statistics_match_upstream_thrml(parity_results):
    diffs = parity_results["mixed"]["diffs"]
    assert diffs["discrete_hist_linf"] <= 0.05
    assert diffs["gaussian_mean_linf"] <= 1.0e-3
    assert diffs["gaussian_cov_linf"] <= 2.0e-3


def test_observer_and_clamped_state_moments_match_upstream_thrml(parity_results):
    diffs = parity_results["observation_clamp"]["diffs"]
    assert diffs["first_moment_linf"] <= 0.02
    assert diffs["second_moment_linf"] <= 0.02
