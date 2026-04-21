from pathlib import Path

try:
    import torch
except ImportError:
    from tests.parity._torch_stub import install_torch_stub

    torch = install_torch_stub()

from tt_thrml.compiler.categorical_ops import CategoricalThetaInputs, dense_categorical_theta_op
from tt_thrml.compiler.discrete_ebm_packing import PackedCategoricalThetaBatch
from tt_thrml.compiler.ttmlir.categorical_theta import (
    categorical_theta_expected_from_batches,
    categorical_theta_op_signature,
    lower_categorical_theta_inputs_to_stablehlo,
    lower_packed_categorical_theta_to_stablehlo,
    make_ttmlir_categorical_theta_op,
)
from tt_thrml.compiler.ttmlir.runtime import (
    TTMLIRCompiledArtifact,
    TTMLIRConfig,
    TTMLIRExecutionResult,
)
from tests.ttnn_test_utils import FakeTTNN


def test_ttmlir_categorical_theta_expected_matches_manual_dense_result():
    batch = PackedCategoricalThetaBatch(
        interaction_indices=(0,),
        n_spin=1,
        n_categories=3,
        tail_shape=(2,),
        weights=torch.tensor([[[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]]], dtype=torch.float32),
        active_mask=torch.tensor([[[1.0]]], dtype=torch.float32),
        spin_conditions=(torch.tensor([[[1]]], dtype=torch.int32),),
        categorical_conditions=(torch.tensor([[[1]]], dtype=torch.int64),),
    )

    actual = categorical_theta_expected_from_batches((batch,))
    expected = torch.tensor([[[[10.0, 20.0, 30.0]]]], dtype=torch.float32)

    assert torch.allclose(actual, expected)


def test_ttmlir_categorical_theta_stablehlo_mentions_stablehlo_ops():
    batch = PackedCategoricalThetaBatch(
        interaction_indices=(0,),
        n_spin=1,
        n_categories=3,
        tail_shape=(),
        weights=torch.ones((1, 1, 3), dtype=torch.float32),
        active_mask=torch.ones((1, 1), dtype=torch.float32),
        spin_conditions=(torch.ones((1, 1), dtype=torch.int32),),
        categorical_conditions=(),
    )

    text = lower_packed_categorical_theta_to_stablehlo(batch)

    assert "stablehlo" in text


def test_ttmlir_categorical_theta_input_lowering_handles_multi_node_tail_indices():
    text = lower_categorical_theta_inputs_to_stablehlo(
        flat_weights=torch.ones((1, 64, 5, 5), dtype=torch.float32),
        flat_index=torch.zeros((1, 32, 2, 1), dtype=torch.uint32),
        interaction_scale=torch.ones((1, 64, 1, 1), dtype=torch.float32),
        n_nodes=32,
        n_interactions=2,
        n_categories=5,
    )

    assert "stablehlo" in text


def test_ttmlir_categorical_theta_op_caches_compilation_and_matches_dense_path(
    monkeypatch,
    tmp_path: Path,
):
    fake_ttnn = FakeTTNN()
    inputs = CategoricalThetaInputs(
        flat_weights=fake_ttnn.from_torch(
            torch.tensor(
                [[[[1.0, -0.5, 2.0], [-1.5, 0.75, 0.25], [0.0, 1.5, -0.75]]]],
                dtype=torch.float32,
            ),
            dtype=fake_ttnn.bfloat16,
            layout=fake_ttnn.ROW_MAJOR_LAYOUT,
            device="fake",
        ),
        flat_index=fake_ttnn.from_torch(
            torch.tensor([[[[2]]]], dtype=torch.uint32),
            dtype=fake_ttnn.uint32,
            layout=fake_ttnn.ROW_MAJOR_LAYOUT,
            device="fake",
        ),
        interaction_scale=fake_ttnn.from_torch(
            torch.ones((1, 1, 1, 1), dtype=torch.float32),
            dtype=fake_ttnn.bfloat16,
            layout=fake_ttnn.ROW_MAJOR_LAYOUT,
            device="fake",
        ),
        n_nodes=1,
        n_interactions=1,
        n_categories=3,
    )

    compile_calls = []
    run_calls = []

    monkeypatch.setattr(
        "tt_thrml.compiler.ttmlir.categorical_theta.make_ttmlir_config",
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
        if len(input_tensors) == 3:
            flat_weights, flat_index, interaction_scale = input_tensors
        else:
            flat_weights, interaction_scale = input_tensors
            flat_index = None
        expected = dense_categorical_theta_op(
            ttnn=FakeTTNN(),
            device="fake",
            inputs=CategoricalThetaInputs(
                flat_weights=flat_weights,
                flat_index=flat_index,
                interaction_scale=interaction_scale,
                n_nodes=1,
                n_interactions=1,
                n_categories=3,
            ),
        ).to(torch.float32)
        return TTMLIRExecutionResult(
            outputs=(expected,),
            runtime_output_dtypes=("DataType.Float32",),
        )

    monkeypatch.setattr(
        "tt_thrml.compiler.ttmlir.categorical_theta.compile_stablehlo_to_flatbuffer",
        fake_compile,
    )
    monkeypatch.setattr(
        "tt_thrml.compiler.ttmlir.categorical_theta.run_flatbuffer",
        fake_run,
    )
    monkeypatch.setattr(
        "tt_thrml.compiler.ttmlir.categorical_theta.supports_direct_ttnn_inputs",
        lambda *, device=None: True,
    )

    op_a = make_ttmlir_categorical_theta_op(
        config=TTMLIRConfig(
            system_desc_path=tmp_path / "system_desc.ttsys",
            artifact_root=tmp_path / "artifacts",
        )
    )
    op_b = make_ttmlir_categorical_theta_op(
        config=TTMLIRConfig(
            system_desc_path=tmp_path / "system_desc.ttsys",
            artifact_root=tmp_path / "artifacts",
        )
    )

    result_a = op_a(ttnn=fake_ttnn, device="fake", inputs=inputs)
    result_b = op_b(ttnn=fake_ttnn, device="fake", inputs=inputs)
    expected = dense_categorical_theta_op(ttnn=FakeTTNN(), device="fake", inputs=inputs)

    assert torch.allclose(result_a, expected)
    assert torch.allclose(result_b, expected)
    assert len(compile_calls) == 1
    assert len(run_calls) == 2
    assert fake_ttnn.to_torch_calls == 0


def test_ttmlir_categorical_theta_op_signature_distinguishes_tail_and_dense_shapes():
    dense_signature = categorical_theta_op_signature(
        flat_weights=torch.ones((1, 2, 3, 1), dtype=torch.float32),
        flat_index=None,
        interaction_scale=torch.ones((1, 2, 1, 1), dtype=torch.float32),
        n_nodes=1,
        n_interactions=2,
        n_categories=3,
    )
    tail_signature = categorical_theta_op_signature(
        flat_weights=torch.ones((1, 2, 3, 4), dtype=torch.float32),
        flat_index=torch.zeros((1, 2, 1, 1), dtype=torch.uint32),
        interaction_scale=torch.ones((1, 2, 1, 1), dtype=torch.float32),
        n_nodes=1,
        n_interactions=2,
        n_categories=3,
    )

    assert dense_signature != tail_signature
    assert dense_signature.stable_cache_key() != tail_signature.stable_cache_key()
