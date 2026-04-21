from __future__ import annotations

from dataclasses import dataclass


def _first_available_attr(obj, *names: str, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


@dataclass(frozen=True)
class PhysicalTensorSpec:
    shape_tail: tuple[int, ...]
    layout: object | None
    dtype: object | None

    def shape(self, batch_size: int) -> tuple[int, ...]:
        return (int(batch_size), *self.shape_tail)


def tensor_spec_from_reference(reference, *, fallback_layout=None, fallback_dtype=None) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(
        shape_tail=tuple(int(dim) for dim in tuple(getattr(reference, "shape", ()))[1:]),
        layout=getattr(reference, "layout", fallback_layout),
        dtype=getattr(reference, "dtype", fallback_dtype),
    )


def block_state_tensor_spec(*, n_nodes: int, layout, dtype) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(shape_tail=(1, 1, int(n_nodes)), layout=layout, dtype=dtype)


def flat_storage_tensor_spec(*, n_values: int, layout, dtype) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(shape_tail=(1, 1, int(n_values)), layout=layout, dtype=dtype)


def sampler_state_tensor_spec(*, shape: tuple[int, ...], layout, dtype) -> PhysicalTensorSpec:
    flat_size = 1
    for dim in shape:
        flat_size *= int(dim)
    return flat_storage_tensor_spec(n_values=flat_size, layout=layout, dtype=dtype)


def gathered_source_tensor_spec(
    n_nodes: int,
    n_interactions: int,
    *,
    layout,
    dtype,
) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(
        shape_tail=(int(n_nodes), int(n_interactions), 1),
        layout=layout,
        dtype=dtype,
    )


def grouped_interaction_tensor_spec(
    n_nodes: int,
    n_interactions: int,
    *,
    layout,
    dtype,
) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(
        shape_tail=(1, int(n_nodes), int(n_interactions), 1),
        layout=layout,
        dtype=dtype,
    )


def flattened_interaction_tensor_spec(
    n_nodes: int,
    n_interactions: int,
    *,
    layout,
    dtype,
) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(
        shape_tail=(int(n_nodes) * int(n_interactions), 1, 1),
        layout=layout,
        dtype=dtype,
    )


def interaction_scale_tensor_spec(
    n_nodes: int,
    n_interactions: int,
    *,
    has_tail: bool,
    layout,
    dtype,
) -> PhysicalTensorSpec:
    if has_tail:
        return PhysicalTensorSpec(
            shape_tail=(1, int(n_nodes), int(n_interactions), 1),
            layout=layout,
            dtype=dtype,
        )
    return PhysicalTensorSpec(
        shape_tail=(1, int(n_nodes), int(n_interactions)),
        layout=layout,
        dtype=dtype,
    )


def spin_parameter_tensor_spec(*, n_nodes: int, layout, dtype) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(shape_tail=(1, int(n_nodes), 1), layout=layout, dtype=dtype)


def categorical_parameter_tensor_spec(
    *,
    n_nodes: int,
    n_categories: int,
    layout,
    dtype,
) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(
        shape_tail=(1, int(n_nodes), int(n_categories)),
        layout=layout,
        dtype=dtype,
    )


def gaussian_parameter_tensor_spec(*, n_nodes: int, layout, dtype) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(shape_tail=(1, int(n_nodes), 2), layout=layout, dtype=dtype)


def spin_gaussian_weight_tensor_spec(
    *,
    n_nodes: int,
    n_interactions: int,
    tail_size: int,
    layout,
    dtype,
) -> PhysicalTensorSpec:
    if int(tail_size) > 1:
        return PhysicalTensorSpec(
            shape_tail=(1, int(n_nodes), int(n_interactions), int(tail_size)),
            layout=layout,
            dtype=dtype,
        )
    return PhysicalTensorSpec(
        shape_tail=(1, int(n_nodes), int(n_interactions)),
        layout=layout,
        dtype=dtype,
    )


def categorical_weight_tensor_spec(
    *,
    n_nodes: int,
    n_interactions: int,
    n_categories: int,
    tail_size: int,
    layout,
    dtype,
) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(
        shape_tail=(int(n_nodes) * int(n_interactions), int(n_categories), int(tail_size)),
        layout=layout,
        dtype=dtype,
    )


def categorical_weight_group_tensor_spec(
    *,
    n_nodes: int,
    n_interactions: int,
    n_categories: int,
    tail_size: int,
    layout,
    dtype,
) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(
        shape_tail=(int(n_nodes), int(n_interactions), int(n_categories), int(tail_size)),
        layout=layout,
        dtype=dtype,
    )


def categorical_active_mask_tensor_spec(
    *,
    n_nodes: int,
    n_interactions: int,
    layout,
    dtype,
) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(
        shape_tail=(int(n_nodes) * int(n_interactions), 1, 1),
        layout=layout,
        dtype=dtype,
    )


def categorical_active_mask_group_tensor_spec(
    *,
    n_nodes: int,
    n_interactions: int,
    layout,
    dtype,
) -> PhysicalTensorSpec:
    return PhysicalTensorSpec(
        shape_tail=(int(n_nodes), int(n_interactions), 1, 1),
        layout=layout,
        dtype=dtype,
    )


__all__ = [
    "PhysicalTensorSpec",
    "_first_available_attr",
    "block_state_tensor_spec",
    "categorical_active_mask_group_tensor_spec",
    "categorical_active_mask_tensor_spec",
    "categorical_parameter_tensor_spec",
    "categorical_weight_group_tensor_spec",
    "categorical_weight_tensor_spec",
    "flat_storage_tensor_spec",
    "flattened_interaction_tensor_spec",
    "gathered_source_tensor_spec",
    "gaussian_parameter_tensor_spec",
    "grouped_interaction_tensor_spec",
    "interaction_scale_tensor_spec",
    "sampler_state_tensor_spec",
    "spin_gaussian_weight_tensor_spec",
    "spin_parameter_tensor_spec",
    "tensor_spec_from_reference",
]
