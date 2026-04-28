"""Minimal TT-Lang hardware hang for simultaneous carried dataflow buffers.

Observed with TT-Lang 1.0.0 on Wormhole:

- simulator: passes
- hardware: hangs during execution

Run from the TT-Lang tooling environment:

    /Users/ludwig/.claude/commands/tools/run-test.sh /abs/path/to/this/file.py
    /Users/ludwig/.claude/commands/tools/run-test.sh --hw /abs/path/to/this/file.py

If the hardware run hangs, kill the remote Python test process and reset the
board from the TT-Lang container:

    podman exec tt-lang-codex bash -lc "tt-smi -r all --no_reinit"

The sequential version of this pattern is hardware-clean. The hang appears when
several old CB fronts are held live while reserving and pushing the next fronts
for those same carried lanes.
"""

from __future__ import annotations

import torch
import ttl
import ttnn

TILE = 32
LANES = 4


def to_tt(tensor: torch.Tensor, device: object) -> object:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@ttl.operation(grid=(1, 1))
def simultaneous_carry(seed, increments, out):
    c0 = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    c1 = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    c2 = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    c3 = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    i0 = ttl.make_dataflow_buffer_like(increments, shape=(1, 1), block_count=2)
    i1 = ttl.make_dataflow_buffer_like(increments, shape=(1, 1), block_count=2)
    i2 = ttl.make_dataflow_buffer_like(increments, shape=(1, 1), block_count=2)
    i3 = ttl.make_dataflow_buffer_like(increments, shape=(1, 1), block_count=2)
    o0 = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    o1 = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    o2 = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    o3 = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        a0 = c0.wait()
        a1 = c1.wait()
        a2 = c2.wait()
        a3 = c3.wait()
        b0 = i0.wait()
        b1 = i1.wait()
        b2 = i2.wait()
        b3 = i3.wait()

        n0 = c0.reserve()
        n0.store(a0 + b0)
        n0.push()
        n1 = c1.reserve()
        n1.store(a1 + b1)
        n1.push()
        n2 = c2.reserve()
        n2.store(a2 + b2)
        n2.push()
        n3 = c3.reserve()
        n3.store(a3 + b3)
        n3.push()

        b3.pop()
        b2.pop()
        b1.pop()
        b0.pop()
        a3.pop()
        a2.pop()
        a1.pop()
        a0.pop()

        f0 = c0.wait()
        out0 = o0.reserve()
        out0.store(f0)
        out0.push()
        f0.pop()
        f1 = c1.wait()
        out1 = o1.reserve()
        out1.store(f1)
        out1.push()
        f1.pop()
        f2 = c2.wait()
        out2 = o2.reserve()
        out2.store(f2)
        out2.push()
        f2.pop()
        f3 = c3.wait()
        out3 = o3.reserve()
        out3.store(f3)
        out3.push()
        f3.pop()

    @ttl.datamovement()
    def read():
        s0 = c0.reserve()
        ttl.copy(seed[0, 0], s0).wait()
        s0.push()
        s1 = c1.reserve()
        ttl.copy(seed[1, 0], s1).wait()
        s1.push()
        s2 = c2.reserve()
        ttl.copy(seed[2, 0], s2).wait()
        s2.push()
        s3 = c3.reserve()
        ttl.copy(seed[3, 0], s3).wait()
        s3.push()

        x0 = i0.reserve()
        ttl.copy(increments[0, 0], x0).wait()
        x0.push()
        x1 = i1.reserve()
        ttl.copy(increments[1, 0], x1).wait()
        x1.push()
        x2 = i2.reserve()
        ttl.copy(increments[2, 0], x2).wait()
        x2.push()
        x3 = i3.reserve()
        ttl.copy(increments[3, 0], x3).wait()
        x3.push()

    @ttl.datamovement()
    def write():
        x0 = o0.wait()
        ttl.copy(x0, out[0, 0]).wait()
        x0.pop()
        x1 = o1.wait()
        ttl.copy(x1, out[1, 0]).wait()
        x1.pop()
        x2 = o2.wait()
        ttl.copy(x2, out[2, 0]).wait()
        x2.pop()
        x3 = o3.wait()
        ttl.copy(x3, out[3, 0]).wait()
        x3.pop()


def make_planes(values: list[float]) -> torch.Tensor:
    return torch.stack([torch.full((TILE, TILE), value, dtype=torch.bfloat16) for value in values]).reshape(
        len(values) * TILE,
        TILE,
    )


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        seed = make_planes([0.0, 1.0, 2.0, 3.0])
        increments = make_planes([10.0, 11.0, 12.0, 13.0])
        out = to_tt(torch.zeros((LANES * TILE, TILE), dtype=torch.bfloat16), device)

        simultaneous_carry(to_tt(seed, device), to_tt(increments, device), out)

        result = ttnn.to_torch(out).to(torch.float32)
        expected = (seed + increments).to(torch.float32)
        print("result", [float(result[index * TILE, 0]) for index in range(LANES)])
        print("expected", [float(expected[index * TILE, 0]) for index in range(LANES)])
        assert torch.allclose(result, expected, rtol=0, atol=0)
        print("PASS: simultaneous carry")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
