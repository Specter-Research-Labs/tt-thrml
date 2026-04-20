from __future__ import annotations

from types import SimpleNamespace

import torch


class FakeTTNN:
    bfloat16 = torch.float32
    uint32 = torch.int64
    int32 = torch.int64
    ROW_MAJOR_LAYOUT = "row_major"
    TILE_LAYOUT = "tile"

    def __init__(self):
        self.begin_trace_calls = 0
        self.end_trace_calls = 0
        self.execute_trace_calls = 0
        self.release_trace_calls = 0
        self.experimental = SimpleNamespace()
        self.python_operation_registrations = 0
        self.sampling_calls = 0
        self.synchronize_calls = 0
        self.to_layout_calls = 0
        self.to_torch_calls = 0
        self.sum_calls = 0
        self.repeat_calls = 0
        self.gather_calls = 0
        self.multiply_calls = 0
        self.add_calls = 0
        self.reciprocal_calls = 0
        self.sqrt_calls = 0
        self.argmax_calls = 0
        self.gt_calls = 0
        self.where_calls = 0
        self.from_torch_layouts = []

    def from_torch(self, value, *, dtype=None, layout=None, device=None):
        self.from_torch_layouts.append(layout)
        tensor = (
            value.clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        )
        return tensor.to(dtype or tensor.dtype)

    def to_torch(self, value):
        self.to_torch_calls += 1
        return value.clone()

    def full(self, shape, *, fill_value, dtype=None, layout=None, device=None):
        return torch.full(shape, fill_value=fill_value, dtype=dtype or torch.float32)

    def repeat(self, value, sizes):
        self.repeat_calls += 1
        return value.repeat(sizes)

    def concat(self, values, dim=0):
        return torch.concat(values, dim=dim)

    def gather(self, values, dim, *, index):
        self.gather_calls += 1
        return torch.gather(values, dim, index.to(torch.int64))

    def multiply(self, lhs, rhs):
        self.multiply_calls += 1
        return lhs * rhs

    def sum(self, value, *, dim, keepdim):
        self.sum_calls += 1
        return torch.sum(value, dim=dim, keepdim=keepdim)

    def add(self, lhs, rhs):
        self.add_calls += 1
        return lhs + rhs

    def reciprocal(self, value):
        self.reciprocal_calls += 1
        return torch.reciprocal(value)

    def sqrt(self, value):
        self.sqrt_calls += 1
        return torch.sqrt(value)

    def argmax(self, value, dim=None, keepdim=False, **kwargs):
        del kwargs
        self.argmax_calls += 1
        return torch.argmax(value, dim=dim, keepdim=keepdim)

    def gt(self, lhs, rhs):
        self.gt_calls += 1
        return (lhs > rhs).to(lhs.dtype)

    def sigmoid(self, value):
        return torch.sigmoid(value)

    def bernoulli(self, value, seed=None):
        del seed
        return (value >= 0.5).to(value.dtype)

    def where(self, condition, lhs, rhs):
        self.where_calls += 1
        return torch.where(condition.to(torch.bool), lhs, rhs)

    def reshape(self, value, shape):
        return torch.reshape(value, shape)

    def to_layout(self, value, layout):
        del layout
        self.to_layout_calls += 1
        return value

    def typecast(self, value, *, dtype):
        return value.to(dtype)

    def to_dtype(self, value, dtype):
        return value.to(dtype)

    def register_python_operation(
        self,
        *,
        name,
        is_experimental=False,
        is_method=False,
        golden_function=None,
        preprocess_golden_function_inputs=None,
        postprocess_golden_function_outputs=None,
        doc=None,
    ):
        del (
            is_method,
            golden_function,
            preprocess_golden_function_inputs,
            postprocess_golden_function_outputs,
            doc,
        )

        def decorator(function):
            self.python_operation_registrations += 1

            if not name.startswith("ttnn."):
                raise RuntimeError("Expected TTNN operation name to start with 'ttnn.'")

            target = self
            for segment in name.split(".")[1:-1]:
                current = getattr(target, segment, None)
                if current is None:
                    current = SimpleNamespace()
                    setattr(target, segment, current)
                target = current

            setattr(target, name.split(".")[-1], function)
            return function

        return decorator

    def sampling(
        self,
        input_values_tensor,
        input_indices_tensor,
        *,
        k,
        p,
        temp,
        seed=None,
        sub_core_grids=None,
        output_tensor=None,
    ):
        del p, temp, seed, sub_core_grids, output_tensor
        self.sampling_calls += 1

        values = input_values_tensor.to(torch.float32).reshape(
            -1, input_values_tensor.shape[-1]
        )
        indices = input_indices_tensor.to(torch.int64).reshape(
            -1, input_indices_tensor.shape[-1]
        )
        k_values = k.to(torch.int64).reshape(-1)

        outputs = []
        for row_values, row_indices, row_k in zip(values, indices, k_values):
            topk = max(int(row_k.item()), 1)
            _, topk_positions = torch.topk(row_values, k=topk, dim=-1)
            outputs.append(row_indices[topk_positions[0]].to(torch.int64))

        return torch.stack(outputs).reshape(1, 1, 1, -1)

    def begin_trace_capture(self, device, *, cq_id=0):
        self.begin_trace_calls += 1
        return self.begin_trace_calls

    def end_trace_capture(self, device, trace_id, *, cq_id=0):
        self.end_trace_calls += 1

    def execute_trace(self, device, trace_id, *, cq_id=0, blocking=True):
        self.execute_trace_calls += 1

    def release_trace(self, device, trace_id):
        self.release_trace_calls += 1

    def synchronize_device(self, device):
        self.synchronize_calls += 1
