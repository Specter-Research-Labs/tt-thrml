from __future__ import annotations

import importlib.machinery
import sys
import types

import numpy as np


def _coerce_array(value) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.array
    return np.asarray(value)


class Tensor:
    __array_priority__ = 1000

    def __init__(self, value):
        self.array = np.asarray(value)

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def ndim(self):
        return self.array.ndim

    def clone(self):
        return Tensor(self.array.copy())

    def to(self, dtype=None, *_, **kwargs):
        requested = kwargs.get("dtype", dtype)
        if requested is None:
            return Tensor(self.array.copy())
        return Tensor(self.array.astype(requested))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(self.array.reshape(*shape))

    def repeat(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        return Tensor(np.tile(self.array, tuple(int(size) for size in sizes)))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.array)

    def tolist(self):
        return self.array.tolist()

    def item(self):
        return self.array.item()

    def squeeze(self, dim=None):
        return Tensor(np.squeeze(self.array, axis=dim))

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.array, axis=dim))

    def expand(self, *shape):
        return Tensor(np.broadcast_to(self.array, shape))

    def contiguous(self):
        return self

    def flatten(self):
        return Tensor(self.array.reshape(-1))

    def astype(self, dtype):
        return Tensor(self.array.astype(dtype))

    def float(self):
        return Tensor(self.array.astype(np.float32))

    def int(self):
        return Tensor(self.array.astype(np.int32))

    def long(self):
        return Tensor(self.array.astype(np.int64))

    def bool(self):
        return Tensor(self.array.astype(np.bool_))

    def is_floating_point(self):
        return bool(np.issubdtype(self.array.dtype, np.floating))

    def __array__(self, dtype=None):
        if dtype is None:
            return np.asarray(self.array)
        return np.asarray(self.array, dtype=dtype)

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        for value in self.array:
            yield Tensor(value)

    def __getitem__(self, item):
        return Tensor(self.array[item])

    def __repr__(self):
        return f"Tensor({self.array!r})"

    def _binary(self, other, op):
        return Tensor(op(self.array, _coerce_array(other)))

    def __add__(self, other):
        return self._binary(other, np.add)

    def __radd__(self, other):
        return Tensor(np.add(_coerce_array(other), self.array))

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    def __rsub__(self, other):
        return Tensor(np.subtract(_coerce_array(other), self.array))

    def __mul__(self, other):
        return self._binary(other, np.multiply)

    def __rmul__(self, other):
        return Tensor(np.multiply(_coerce_array(other), self.array))

    def __truediv__(self, other):
        return self._binary(other, np.divide)

    def __rtruediv__(self, other):
        return Tensor(np.divide(_coerce_array(other), self.array))

    def __neg__(self):
        return Tensor(-self.array)

    def __gt__(self, other):
        return Tensor(np.greater(self.array, _coerce_array(other)))


def install_torch_stub():
    existing = sys.modules.get("torch")
    if existing is not None:
        return existing

    torch = types.ModuleType("torch")
    torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    torch._TT_THRML_TORCH_STUB = True
    torch.Tensor = Tensor
    torch.float32 = np.float32
    torch.float16 = np.float16
    torch.bfloat16 = np.float32
    torch.int64 = np.int64
    torch.int32 = np.int32
    torch.uint32 = np.uint32
    torch.uint16 = np.uint16
    torch.uint8 = np.uint8
    torch.bool = np.bool_

    torch.from_numpy = lambda value: Tensor(np.array(value, copy=True))
    torch.tensor = lambda value, dtype=None, **_: Tensor(
        np.array(value, copy=True, dtype=dtype)
    )
    torch.as_tensor = lambda value: Tensor(_coerce_array(value))
    torch.empty = lambda shape, dtype=np.float32: Tensor(np.empty(shape, dtype=dtype))
    torch.empty_like = lambda value, dtype=None, **_: Tensor(
        np.empty_like(_coerce_array(value), dtype=dtype)
    )

    def _shape_args(shape_like):
        if isinstance(shape_like, (tuple, list)):
            return tuple(int(value) for value in shape_like)
        return (int(shape_like),)

    torch.zeros = lambda *shape, dtype=np.float32, **_: Tensor(
        np.zeros(
            _shape_args(shape[0]) if len(shape) == 1 else tuple(int(value) for value in shape),
            dtype=dtype,
        )
    )
    torch.full = lambda shape, fill_value, dtype=np.float32, **_: Tensor(
        np.full(_shape_args(shape), fill_value, dtype=dtype)
    )
    torch.full_like = lambda value, fill_value, dtype=None, **_: Tensor(
        np.full_like(_coerce_array(value), fill_value, dtype=dtype)
    )
    torch.ones = lambda *shape, dtype=np.float32, **_: Tensor(
        np.ones(
            _shape_args(shape[0]) if len(shape) == 1 else tuple(int(value) for value in shape),
            dtype=dtype,
        )
    )
    torch.arange = lambda *args, dtype=np.int64, **_: Tensor(np.arange(*args, dtype=dtype))
    torch.range = torch.arange
    torch.linspace = lambda start, end, steps, **_: Tensor(
        np.linspace(start, end, int(steps), dtype=np.float32)
    )
    torch.logspace = lambda start, end, steps, base=10.0, **_: Tensor(
        np.logspace(start, end, int(steps), base=base, dtype=np.float32)
    )
    torch.eye = lambda n, m=None, dtype=np.float32, **_: Tensor(
        np.eye(int(n), int(n if m is None else m), dtype=dtype)
    )
    torch.reshape = lambda value, shape: Tensor(np.reshape(_coerce_array(value), shape))
    torch.concat = lambda values, dim=0: Tensor(
        np.concatenate([_coerce_array(value) for value in values], axis=dim)
    )
    torch.stack = lambda values, dim=0: Tensor(
        np.stack([_coerce_array(value) for value in values], axis=dim)
    )
    torch.gather = lambda values, dim, index: Tensor(
        np.take_along_axis(
            _coerce_array(values),
            _coerce_array(index).astype(np.int64),
            axis=dim,
        )
    )
    torch.sum = lambda value, dim=None, keepdim=False: Tensor(
        np.sum(_coerce_array(value), axis=dim, keepdims=keepdim)
    )
    torch.where = lambda condition, lhs, rhs: Tensor(
        np.where(
            _coerce_array(condition).astype(np.bool_),
            _coerce_array(lhs),
            _coerce_array(rhs),
        )
    )
    torch.ones_like = lambda value, dtype=None, **_: Tensor(
        np.ones_like(_coerce_array(value), dtype=dtype)
    )
    torch.zeros_like = lambda value, dtype=None, **_: Tensor(
        np.zeros_like(_coerce_array(value), dtype=dtype)
    )
    torch.rand = lambda *shape, **_: Tensor(
        np.random.rand(
            *(_shape_args(shape[0]) if len(shape) == 1 else tuple(int(value) for value in shape))
        ).astype(np.float32)
    )
    torch.rand_like = lambda value, **_: Tensor(np.random.rand(*_coerce_array(value).shape).astype(np.float32))
    torch.randn = lambda *shape, **_: Tensor(
        np.random.randn(
            *(_shape_args(shape[0]) if len(shape) == 1 else tuple(int(value) for value in shape))
        ).astype(np.float32)
    )
    torch.randn_like = lambda value, **_: Tensor(np.random.randn(*_coerce_array(value).shape).astype(np.float32))
    torch.randint = lambda low, high, size, dtype=np.int64, **_: Tensor(
        np.random.randint(low, high, size=_shape_args(size), dtype=dtype)
    )
    torch.randint_like = lambda value, low, high, dtype=None, **_: Tensor(
        np.random.randint(
            low,
            high,
            size=_coerce_array(value).shape,
            dtype=np.int64 if dtype is None else dtype,
        )
    )
    torch.randperm = lambda n, **_: Tensor(np.random.permutation(int(n)).astype(np.int64))
    torch.bernoulli = lambda value, **_: Tensor(
        np.random.binomial(1, _coerce_array(value)).astype(np.float32)
    )
    torch.multinomial = lambda input, num_samples, replacement=False, **_: Tensor(
        np.random.choice(
            np.arange(_coerce_array(input).shape[-1]),
            size=int(num_samples),
            replace=bool(replacement),
        ).astype(np.int64)
    )
    torch.normal = lambda mean=0.0, std=1.0, size=None, **_: Tensor(
        np.random.normal(mean, std, size=None if size is None else _shape_args(size)).astype(np.float32)
    )
    torch.poisson = lambda value, **_: Tensor(np.random.poisson(_coerce_array(value)).astype(np.float32))
    torch.complex = lambda real, imag, **_: Tensor(
        _coerce_array(real).astype(np.complex64) + 1j * _coerce_array(imag).astype(np.complex64)
    )
    torch.heaviside = lambda input, values, **_: Tensor(
        np.heaviside(_coerce_array(input), _coerce_array(values))
    )

    def _argmax(value, dim=None, keepdim=False, **_):
        result = np.argmax(_coerce_array(value), axis=dim)
        if keepdim and dim is not None:
            result = np.expand_dims(result, axis=dim)
        return Tensor(result)

    torch.argmax = _argmax
    torch.allclose = lambda lhs, rhs, rtol=1e-05, atol=1e-08, equal_nan=False: bool(
        np.allclose(
            _coerce_array(lhs),
            _coerce_array(rhs),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
    )
    torch.equal = lambda lhs, rhs: bool(
        np.array_equal(_coerce_array(lhs), _coerce_array(rhs))
    )

    def _topk(value, k, dim=-1, largest=True, sorted=True, **_):
        del sorted
        arr = _coerce_array(value)
        k = int(k)
        axis = int(dim)
        if largest:
            partition = np.argpartition(-arr, kth=k - 1, axis=axis)
            indices = np.take(partition, indices=np.arange(k), axis=axis)
            values = np.take_along_axis(arr, indices, axis=axis)
            order = np.argsort(-values, axis=axis)
        else:
            partition = np.argpartition(arr, kth=k - 1, axis=axis)
            indices = np.take(partition, indices=np.arange(k), axis=axis)
            values = np.take_along_axis(arr, indices, axis=axis)
            order = np.argsort(values, axis=axis)
        indices = np.take_along_axis(indices, order, axis=axis)
        values = np.take_along_axis(values, order, axis=axis)
        return Tensor(values), Tensor(indices.astype(np.int64))

    torch.topk = _topk
    torch.reciprocal = lambda value: Tensor(np.reciprocal(_coerce_array(value)))
    torch.sqrt = lambda value: Tensor(np.sqrt(_coerce_array(value)))
    torch.sigmoid = lambda value: Tensor(1.0 / (1.0 + np.exp(-_coerce_array(value))))

    class _Module:
        def __call__(self, *args, **kwargs):
            forward = getattr(self, "forward", None)
            if callable(forward):
                return forward(*args, **kwargs)
            raise NotImplementedError("Torch stub module has no forward().")

    class _Parameter(Tensor):
        pass

    class _Functional(types.SimpleNamespace):
        @staticmethod
        def one_hot(index, num_classes):
            arr = _coerce_array(index).astype(np.int64)
            eye = np.eye(int(num_classes), dtype=np.int64)
            return Tensor(eye[arr])

    torch.nn = types.SimpleNamespace(
        Module=_Module,
        Parameter=_Parameter,
        functional=_Functional(),
    )

    sys.modules["torch"] = torch
    return torch


__all__ = ["Tensor", "install_torch_stub"]
