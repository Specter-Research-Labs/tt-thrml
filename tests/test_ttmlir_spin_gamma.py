from pathlib import Path

import torch

from tt_thrml.compiler.discrete_ebm_packing import PackedSpinGammaBatch
from tt_thrml.compiler.spin_ops import SpinGammaInputs, dense_spin_gamma_op
from tt_thrml.compiler.ttmlir.runtime import (
    TTMLIRCompiledArtifact,
    TTMLIRConfig,
    TTMLIRExecutionResult,
)
from tt_thrml.compiler.ttmlir.spin_gamma import (
    lower_packed_spin_gamma_to_stablehlo,
    lower_spin_gamma_inputs_to_stablehlo,
    make_ttmlir_spin_gamma_op,
    spin_gamma_expected_from_batches,
    spin_gamma_op_signature,
)
from tests.ttnn_test_utils import FakeTTNN


def test_ttmlir_spin_gamma_batch_expected_matches_manual_dense_result():
    batch = PackedSpinGammaBatch(
        interaction_indices=(0,),
        n_spin=1,
        tail_shape=(3,),
        weights=torch.tensor([[[[1.0, -2.0, 0.5]]]], dtype=torch.float32),
        active_mask=torch.tensor([[[1.0]]], dtype=torch.float32),
        spin_conditions=(torch.tensor([[[1]]], dtype=torch.int32),),
        categorical_conditions=(torch.tensor([[[2]]], dtype=torch.int64),),
    )

    actual = spin_gamma_expected_from_batches((batch,))
    expected = torch.tensor([[[[0.5]]]], dtype=torch.float32)

    assert torch.allclose(actual, expected)


def test_ttmlir_spin_gamma_stablehlo_mentions_stablehlo_ops():
    batch = PackedSpinGammaBatch(
        interaction_indices=(0,),
        n_spin=1,
        tail_shape=(),
        weights=torch.ones((1, 1, 1), dtype=torch.float32),
        active_mask=torch.ones((1, 1), dtype=torch.float32),
        spin_conditions=(torch.ones((1, 1), dtype=torch.int32),),
        categorical_conditions=(),
    )

    packed_text = lower_packed_spin_gamma_to_stablehlo(batch)
    inputs_text = lower_spin_gamma_inputs_to_stablehlo(
        flat_weights=torch.ones((1, 1, 1, 1), dtype=torch.float32),
        flat_index=None,
        interaction_scale=torch.ones((1, 1, 1, 1), dtype=torch.float32),
        n_nodes=1,
        n_interactions=1,
    )

    assert "stablehlo" in packed_text
    assert "stablehlo" in inputs_text


def test_ttmlir_spin_gamma_tail_lowering_avoids_stablehlo_gather():
    batch = PackedSpinGammaBatch(
        interaction_indices=(0,),
        n_spin=1,
        tail_shape=(3,),
        weights=torch.tensor([[[[1.0, -2.0, 0.5]]]], dtype=torch.float32),
        active_mask=torch.tensor([[[1.0]]], dtype=torch.float32),
        spin_conditions=(torch.tensor([[[1]]], dtype=torch.int32),),
        categorical_conditions=(torch.tensor([[[2]]], dtype=torch.uint32),),
    )

    packed_text = lower_packed_spin_gamma_to_stablehlo(batch)

    assert "stablehlo.gather" not in packed_text


def test_ttmlir_spin_gamma_op_caches_compilation_and_matches_dense_path(
    monkeypatch,
    tmp_path: Path,
):
    fake_ttnn = FakeTTNN()
    inputs = SpinGammaInputs(
        flat_weights=fake_ttnn.from_torch(
            torch.tensor([[[[[1.0, -0.5, 2.0]]]]], dtype=torch.float32),
            dtype=fake_ttnn.bfloat16,
            layout=fake_ttnn.TILE_LAYOUT,
            device="fake",
        ),
        flat_index=fake_ttnn.from_torch(
            torch.tensor([[[[[2]]]]], dtype=torch.uint32),
            dtype=fake_ttnn.uint32,
            layout=fake_ttnn.ROW_MAJOR_LAYOUT,
            device="fake",
        ),
        interaction_scale=fake_ttnn.from_torch(
            torch.ones((1, 1, 1, 1, 1), dtype=torch.float32),
            dtype=fake_ttnn.bfloat16,
            layout=fake_ttnn.ROW_MAJOR_LAYOUT,
            device="fake",
        ),
        n_nodes=1,
        n_interactions=1,
    )

    compile_calls = []
    run_calls = []

    monkeypatch.setattr(
        "tt_thrml.compiler.ttmlir.spin_gamma.make_ttmlir_config",
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
        expected = dense_spin_gamma_op(
            ttnn=FakeTTNN(),
            device="fake",
            inputs=SpinGammaInputs(
                flat_weights=flat_weights,
                flat_index=flat_index,
                interaction_scale=interaction_scale,
                n_nodes=1,
                n_interactions=1,
            ),
        ).to(torch.float32)
        return TTMLIRExecutionResult(
            outputs=(expected,),
            runtime_output_dtypes=("DataType.Float32",),
        )

    monkeypatch.setattr(
        "tt_thrml.compiler.ttmlir.spin_gamma.compile_stablehlo_to_flatbuffer",
        fake_compile,
    )
    monkeypatch.setattr("tt_thrml.compiler.ttmlir.spin_gamma.run_flatbuffer", fake_run)

    op_a = make_ttmlir_spin_gamma_op(
        config=TTMLIRConfig(
            system_desc_path=tmp_path / "system_desc.ttsys",
            artifact_root=tmp_path / "artifacts",
        )
    )
    op_b = make_ttmlir_spin_gamma_op(
        config=TTMLIRConfig(
            system_desc_path=tmp_path / "system_desc.ttsys",
            artifact_root=tmp_path / "artifacts",
        )
    )

    result_a = op_a(ttnn=fake_ttnn, device="fake", inputs=inputs)
    result_b = op_b(ttnn=fake_ttnn, device="fake", inputs=inputs)
    expected = dense_spin_gamma_op(ttnn=FakeTTNN(), device="fake", inputs=inputs)

    assert torch.allclose(result_a, expected)
    assert torch.allclose(result_b, expected)
    assert len(compile_calls) == 1
    assert len(run_calls) == 2
    assert fake_ttnn.to_torch_calls == 6


def test_ttmlir_spin_gamma_op_passes_original_inputs_to_runtime_when_bridge_supports_ttnn_tensors(
    monkeypatch,
    tmp_path: Path,
):
    fake_ttnn = FakeTTNN()
    inputs = SpinGammaInputs(
        flat_weights=fake_ttnn.from_torch(
            torch.tensor([[[[[1.0, -0.5, 2.0]]]]], dtype=torch.float32),
            dtype=fake_ttnn.bfloat16,
            layout=fake_ttnn.TILE_LAYOUT,
            device="fake",
        ),
        flat_index=fake_ttnn.from_torch(
            torch.tensor([[[[[2]]]]], dtype=torch.uint32),
            dtype=fake_ttnn.uint32,
            layout=fake_ttnn.ROW_MAJOR_LAYOUT,
            device="fake",
        ),
        interaction_scale=fake_ttnn.from_torch(
            torch.ones((1, 1, 1, 1, 1), dtype=torch.float32),
            dtype=fake_ttnn.bfloat16,
            layout=fake_ttnn.ROW_MAJOR_LAYOUT,
            device="fake",
        ),
        n_nodes=1,
        n_interactions=1,
    )

    def fake_compile(paths, *, stablehlo_module_text, artifact_dir, base_name):
        del paths, stablehlo_module_text
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stablehlo_path = artifact_dir / f"{base_name}.stablehlo.mlir"
        ttir_path = artifact_dir / f"{base_name}.ttir.mlir"
        ttnn_path = artifact_dir / f"{base_name}.ttnn.mlir"
        flatbuffer_path = artifact_dir / f"{base_name}.ttnn"
        stablehlo_path.write_text("// stablehlo")
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
        assert input_tensors[0] is inputs.flat_weights
        assert input_tensors[1] is inputs.flat_index
        assert input_tensors[2] is inputs.interaction_scale
        return TTMLIRExecutionResult(
            outputs=(torch.ones((1, 1, 1, 1), dtype=torch.float32),),
            runtime_output_dtypes=("DataType.Float32",),
        )

    monkeypatch.setattr(
        "tt_thrml.compiler.ttmlir.spin_gamma.compile_stablehlo_to_flatbuffer",
        fake_compile,
    )
    monkeypatch.setattr("tt_thrml.compiler.ttmlir.spin_gamma.run_flatbuffer", fake_run)
    monkeypatch.setattr(
        "tt_thrml.compiler.ttmlir.spin_gamma.supports_direct_ttnn_inputs",
        lambda *, device=None: True,
    )

    op = make_ttmlir_spin_gamma_op(
        config=TTMLIRConfig(
            system_desc_path=tmp_path / "system_desc.ttsys",
            artifact_root=tmp_path / "artifacts",
        )
    )

    result = op(ttnn=fake_ttnn, device="fake", inputs=inputs)

    assert torch.allclose(result, torch.ones((1, 1, 1, 1), dtype=torch.float32))
    assert fake_ttnn.to_torch_calls == 3


def test_ttmlir_spin_gamma_op_signature_distinguishes_tail_and_dense_shapes():
    dense_signature = spin_gamma_op_signature(
        flat_weights=torch.ones((1, 1, 2, 1), dtype=torch.float32),
        flat_index=None,
        interaction_scale=torch.ones((1, 1, 2, 1), dtype=torch.float32),
        n_nodes=1,
        n_interactions=2,
    )
    tail_signature = spin_gamma_op_signature(
        flat_weights=torch.ones((1, 1, 2, 4), dtype=torch.float32),
        flat_index=torch.zeros((1, 1, 1, 2, 1), dtype=torch.uint32),
        interaction_scale=torch.ones((1, 1, 1, 2, 1), dtype=torch.float32),
        n_nodes=1,
        n_interactions=2,
    )

    assert dense_signature != tail_signature
    assert dense_signature.stable_cache_key() != tail_signature.stable_cache_key()
