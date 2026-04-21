from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

import tt_thrml


def _missing_module_error(module_name: str) -> ModuleNotFoundError:
    exc = ModuleNotFoundError(f"No module named {module_name!r}")
    exc.name = module_name
    return exc


@pytest.mark.parametrize(
    ("missing_module", "attr_name", "expected_extra"),
    [
        ("jax", "sample_states", "tt-thrml[runtime]"),
        ("torch", "TTMLIRConfig", "tt-thrml[torch]"),
    ],
)
def test_public_lazy_imports_raise_actionable_optional_dependency_errors(
    monkeypatch: pytest.MonkeyPatch,
    missing_module: str,
    attr_name: str,
    expected_extra: str,
) -> None:
    def fake_import_module(module_name: str, package: str | None = None):
        raise _missing_module_error(missing_module)

    monkeypatch.delitem(tt_thrml.__dict__, attr_name, raising=False)
    monkeypatch.setattr(tt_thrml, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        getattr(tt_thrml, attr_name)

    message = str(exc_info.value)
    assert f"`tt_thrml.{attr_name}` requires optional dependency `{missing_module}`" in message
    assert expected_extra in message


def test_pyproject_exposes_runtime_optional_dependency_groups() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "jax" in optional_deps
    assert "torch" in optional_deps
    assert "runtime" in optional_deps
    assert "jax>=0.6" in optional_deps["runtime"]
    assert "torch>=2.6" in optional_deps["runtime"]


def test_repo_has_license_file() -> None:
    assert Path("LICENSE").is_file()
