import torch

from tt_thrml.compiler.ttnn_kernels import (
    categorical_theta_dense_expected,
    select_last_dim_expected,
    spin_gamma_dense_expected,
)


def test_spin_gamma_dense_expected_matches_manual_result():
    weights = torch.tensor(
        [[[[0.5, -1.0, 0.25], [1.5, 0.0, -0.5]]]],
        dtype=torch.float32,
    )
    active_mask = torch.tensor(
        [[[[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]]],
        dtype=torch.float32,
    )
    spin_conditions = [
        torch.tensor([[[[1, 0, 1], [0, 1, 1]]]], dtype=torch.int32),
        torch.tensor([[[[1, 1, 0], [1, 0, 1]]]], dtype=torch.int32),
    ]

    actual = spin_gamma_dense_expected(weights, active_mask, spin_conditions)
    expected = torch.tensor([[[[1.75], [-1.5]]]], dtype=torch.float32)

    assert torch.allclose(actual, expected)


def test_spin_gamma_dense_expected_matches_manual_result_with_categorical_tail():
    weights = torch.tensor(
        [
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        ],
        dtype=torch.float32,
    )
    active_mask = torch.tensor(
        [[[1.0, 0.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    spin_conditions = [torch.tensor([[[1, 0], [0, 1]]], dtype=torch.int32)]
    categorical_conditions = [torch.tensor([[[2, 1], [0, 2]]], dtype=torch.int64)]

    actual = spin_gamma_dense_expected(
        weights, active_mask, spin_conditions, categorical_conditions
    )
    expected = torch.tensor([[[[3.0], [-3.0]]]], dtype=torch.float32)

    assert torch.allclose(actual, expected)


def test_select_last_dim_expected_matches_torch_gather():
    values = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]]],
        dtype=torch.float32,
    )
    index = torch.tensor(
        [[[[3, 1], [0, 2]]]],
        dtype=torch.int64,
    )

    actual = select_last_dim_expected(values, index)
    expected = torch.tensor(
        [[[[4.0, 2.0], [10.0, 30.0]]]],
        dtype=torch.float32,
    )

    assert torch.equal(actual, expected)


def test_categorical_theta_dense_expected_matches_manual_result():
    weights = torch.tensor(
        [
            [
                [
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [10.0, 20.0, 30.0, 40.0],
                        [100.0, 200.0, 300.0, 400.0],
                    ],
                    [
                        [5.0, 6.0, 7.0, 8.0],
                        [50.0, 60.0, 70.0, 80.0],
                        [500.0, 600.0, 700.0, 800.0],
                    ],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    active_mask = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)
    spin_conditions = [torch.tensor([[[1, 0]]], dtype=torch.int32)]
    categorical_conditions = [torch.tensor([[[3, 1]]], dtype=torch.int64)]

    actual = categorical_theta_dense_expected(
        weights, active_mask, spin_conditions, categorical_conditions
    )
    expected = torch.tensor([[[[-2.0, -20.0, -200.0]]]], dtype=torch.float32)

    assert torch.allclose(actual, expected)
