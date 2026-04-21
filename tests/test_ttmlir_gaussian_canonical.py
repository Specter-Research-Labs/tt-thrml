from pathlib import Path

try:
    import torch
except ImportError:
    from tests.parity._torch_stub import install_torch_stub

    torch = install_torch_stub()

from tt_thrml.compiler.gaussian_ops import (
    GaussianCanonicalInputs,
    dense_gaussian_canonical_op,
)
from tt_thrml.compiler.ttmlir.gaussian_canonical import (
    gaussian_canonical_op_signature,
    lower_gaussian_canonical_inputs_to_stablehlo,
    make_ttmlir_gaussian_canonical_op,
)
from tt_thrml.compiler.ttmlir.runtime import (
    TTMLIRCompiledArtifact,
    TTMLIRConfig,
    TTMLIRExecutionResult,
)
from tests.ttnn_test_utils import FakeTTNN


def test_ttmlir_gaussian_canonical_inputs_stablehlo_mentions_stablehlo_ops():
    text = lower_gaussian_canonical_inputs_to_stablehlo(
        flat_weights=torch.ones((1, 1, 2, 3), dtype=torch.float32),
        flat_index=None,
        interaction_scale=torch.ones((1, 1, 2, 3), dtype=torch.float32),
        n_nodes=2,
        n_interactions=3,
        contribution_kind="linear",
    )

    assert "stablehlo" in text


def test_ttmlir_gaussian_canonical_tail_lowering_accepts_four_dim_interaction_scale():
    text = lower_gaussian_canonical_inputs_to_stablehlo(
        flat_weights=torch.ones((1, 1, 8, 5, 5), dtype=torch.float32),
        flat_index=torch.zeros((1, 1, 8, 5, 1), dtype=torch.uint32),
        interaction_scale=torch.ones((1, 1, 8, 5), dtype=torch.float32),
        n_nodes=8,
        n_interactions=5,
        contribution_kind="precision",
    )

    assert "stablehlo" in text


def test_ttmlir_gaussian_canonical_op_caches_compilation_and_matches_dense_path(
    monkeypatch,
    tmp_path: Path,
):
    fake_ttnn = FakeTTNN()
    inputs = GaussianCanonicalInputs(
        flat_weights=fake_ttnn.from_torch(
            torch.tensor([[[[[1.0, -0.5, 2.0]]]]], dtype=torch.float32),
            dtype=fake_ttnn.bfloat16,
            layout=fake_ttnn.TILE_LAYOUT,
            device="fake",
        ),
        flat_index=fake_ttnn.from_torch(
            torch.tensor([[[[[2]]]]], dtype=torch.uint32),
            dtype=fake_ttnn.uint32,
            layout=fake_ttnn.TILE_LAYOUT,
            device="fake",
        ),
        interaction_scale=fake_ttnn.from_torch(
            torch.ones((1, 1, 1, 1, 1), dtype=torch.float32),
            dtype=fake_ttnn.bfloat16,
            layout=fake_ttnn.TILE_LAYOUT,
            device="fake",
        ),
        n_nodes=1,
        n_interactions=3,
        contribution_kind="precision",
    )

    compile_calls = []
    run_calls = []

    monkeypatch.setattr(
        "tt_thrml.compiler.ttmlir.gaussian_canonical.make_ttmlir_config",
        lambda **_: TTMLIRConfig(
            system_desc_path=tmp_path / "system_desc.ttsys",
            artifact_root=tmp_path / "artifacts",
        ),
    )

    def fake_compile(paths, *, stablehlo_module_text, artifact_dir, base_name):
        compile_calls.append((paths, artifact_dir, base_name, stablehlo_module_text))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stablehlo_path = artifact_dir / f"{base_name}.stablehlo.mlir"
        ttir_path = artifact_dir / f"{base_name}.ttir.mlir"
        ttnn_path = artifact_dir / f"{base_name}.ttnn.mlir"
        flatbuffer_path = artifact_dir / f"{base_name}.ttnn"
        stablehlo_path.write_text(stablehlo_module_text)
        ttir_path.write_text("// ttir")
        ttnn_path.write_text("// ttnn")
        flatbuffer_path.write_text("flatbuffer")
        return TTMLIRCompiledArtifact(
            artifact_dir=artifact_dir,
            stablehlo_path=stablehlo_path,
            ttir_path=ttir_path,
            ttnn_path=ttnn_path,
            flatbuffer_path=flatbuffer_path,
            stablehlo_to_ttir_command=("ttmlir-opt",),
            ttir_to_ttnn_command=("ttmlir-opt",),
            ttnn_to_flatbuffer_command=("ttmlir-translate",),
        )

    def fake_run(paths, *, flatbuffer_path, input_tensors, device=None, prefer_device_output=False):
        del paths, flatbuffer_path, device
        assert prefer_device_output is True
        run_calls.append(tuple(tuple(t.shape) for t in input_tensors))
        flat_weights, flat_index, interaction_scale = input_tensors
        expected = dense_gaussian_canonical_op(
            ttnn=FakeTTNN(),
            device="fake",
            inputs=GaussianCanonicalInputs(
                flat_weights=flat_weights,
                flat_index=flat_index,
                interaction_scale=interaction_scale,
                n_nodes=1,
                n_interactions=3,
                contribution_kind="precision",
            ),
        ).to(torch.float32)
        return TTMLIRExecutionResult(
            outputs=(expected,),
            runtime_output_dtypes=("DataType.Float32",),
        )

    monkeypatch.setattr(
        "tt_thrml.compiler.ttmlir.gaussian_canonical.compile_stablehlo_to_flatbuffer",
        fake_compile,
    )
    monkeypatch.setattr(
        "tt_thrml.compiler.ttmlir.gaussian_canonical.run_flatbuffer",
        fake_run,
    )
    monkeypatch.setattr(
        "tt_thrml.compiler.ttmlir.gaussian_canonical.supports_direct_ttnn_inputs",
        lambda *, device=None: True,
    )

    op_a = make_ttmlir_gaussian_canonical_op(
        config=TTMLIRConfig(
            system_desc_path=tmp_path / "system_desc.ttsys",
            artifact_root=tmp_path / "artifacts",
        )
    )
    op_b = make_ttmlir_gaussian_canonical_op(
        config=TTMLIRConfig(
            system_desc_path=tmp_path / "system_desc.ttsys",
            artifact_root=tmp_path / "artifacts",
        )
    )

    result_a = op_a(ttnn=fake_ttnn, device="fake", inputs=inputs)
    result_b = op_b(ttnn=fake_ttnn, device="fake", inputs=inputs)
    expected = dense_gaussian_canonical_op(
        ttnn=FakeTTNN(),
        device="fake",
        inputs=inputs,
    )

    assert torch.allclose(result_a, expected)
    assert torch.allclose(result_b, expected)
    assert len(compile_calls) == 1
    assert len(run_calls) == 2
    assert run_calls == [((1, 1, 1, 1, 3), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1))] * 2
    assert fake_ttnn.to_torch_calls == 0


def test_ttmlir_gaussian_canonical_signature_distinguishes_contribution_kind():
    linear_signature = gaussian_canonical_op_signature(
        flat_weights=torch.ones((1, 1, 2, 3), dtype=torch.float32),
        flat_index=None,
        interaction_scale=torch.ones((1, 1, 2, 3), dtype=torch.float32),
        n_nodes=2,
        n_interactions=3,
        contribution_kind="linear",
    )
    precision_signature = gaussian_canonical_op_signature(
        flat_weights=torch.ones((1, 1, 2, 3), dtype=torch.float32),
        flat_index=torch.zeros((1, 1, 2, 3, 1), dtype=torch.uint32),
        interaction_scale=torch.ones((1, 1, 2, 3), dtype=torch.float32),
        n_nodes=2,
        n_interactions=3,
        contribution_kind="precision",
    )

    assert linear_signature != precision_signature
    assert linear_signature.stable_cache_key() != precision_signature.stable_cache_key()
