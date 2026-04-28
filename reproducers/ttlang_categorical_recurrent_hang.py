"""Minimal categorical recurrent TT-Lang hardware hang.

Observed with TT-Lang 1.0.0 on Wormhole:

- simulator: passes
- hardware: hangs during execution

Run from the TT-Lang tooling environment:

    /Users/ludwig/.claude/commands/tools/run-test.sh /abs/path/to/this/file.py
    /Users/ludwig/.claude/commands/tools/run-test.sh --hw /abs/path/to/this/file.py

This is smaller than `ttlang_coupled_recurrent_hang.py`: it removes the spin
threshold recurrence and keeps only the categorical Gumbel-max recurrence with
writeback into carried category CBs. The one-sweep categorical update used by
the production runtime is hardware-clean; this recurrent carried-CB version is
the reduced whole-window blocker.
"""

from __future__ import annotations

import torch
import ttl
import ttnn

TILE = 32
STEPS = 4
N_CATEGORIES = 3
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
def categorical_recurrent(seed, gumbels, one, half, out):
    spin_dfb = ttl.make_dataflow_buffer_like(seed, shape=(1, 1), block_count=2)
    cat0_dfb = ttl.make_dataflow_buffer_like(seed, shape=(1, 1), block_count=2)
    cat1_dfb = ttl.make_dataflow_buffer_like(seed, shape=(1, 1), block_count=2)
    cat2_dfb = ttl.make_dataflow_buffer_like(seed, shape=(1, 1), block_count=2)
    gumbel_dfb = ttl.make_dataflow_buffer_like(gumbels, shape=(1, 1), block_count=2)
    one_dfb = ttl.make_dataflow_buffer_like(one, shape=(1, 1), block_count=2)
    half_dfb = ttl.make_dataflow_buffer_like(half, shape=(1, 1), block_count=2)
    score0_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    score1_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    score2_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    out0_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    out1_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    out2_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        one_blk = one_dfb.wait()
        half_blk = half_dfb.wait()
        for _ in range(STEPS):
            old_spin = spin_dfb.wait()
            gum0 = gumbel_dfb.wait()
            score0 = score0_dfb.reserve()
            score0.store(gum0 + old_spin)
            score0.push()
            gum0.pop()
            gum1 = gumbel_dfb.wait()
            score1 = score1_dfb.reserve()
            score1.store(gum1 - old_spin)
            score1.push()
            gum1.pop()
            gum2 = gumbel_dfb.wait()
            score2 = score2_dfb.reserve()
            score2.store(gum2)
            score2.push()
            gum2.pop()

            score0_blk = score0_dfb.wait()
            score1_blk = score1_dfb.wait()
            score2_blk = score2_dfb.wait()
            gt01 = (ttl.math.sign(score0_blk - score1_blk) + one_blk) * half_blk
            gt02 = (ttl.math.sign(score0_blk - score2_blk) + one_blk) * half_blk
            old_cat0 = cat0_dfb.wait()
            next_cat0 = cat0_dfb.reserve()
            next_cat0.store(gt01 * gt02 + old_cat0 - old_cat0)
            next_cat0.push()
            old_cat0.pop()

            gt10 = (ttl.math.sign(score1_blk - score0_blk) + one_blk) * half_blk
            gt12 = (ttl.math.sign(score1_blk - score2_blk) + one_blk) * half_blk
            old_cat1 = cat1_dfb.wait()
            next_cat1 = cat1_dfb.reserve()
            next_cat1.store(gt10 * gt12 + old_cat1 - old_cat1)
            next_cat1.push()
            old_cat1.pop()

            gt20 = (ttl.math.sign(score2_blk - score0_blk) + one_blk) * half_blk
            gt21 = (ttl.math.sign(score2_blk - score1_blk) + one_blk) * half_blk
            old_cat2 = cat2_dfb.wait()
            next_cat2 = cat2_dfb.reserve()
            next_cat2.store(gt20 * gt21 + old_cat2 - old_cat2)
            next_cat2.push()
            old_cat2.pop()
            score2_blk.pop()
            score1_blk.pop()
            score0_blk.pop()
            old_spin.pop()

            spin_next = spin_dfb.reserve()
            spin_next.store(one_blk)
            spin_next.push()

        final_cat0 = cat0_dfb.wait()
        final_cat1 = cat1_dfb.wait()
        final_cat2 = cat2_dfb.wait()
        out0 = out0_dfb.reserve()
        out0.store(final_cat0)
        out0.push()
        out1 = out1_dfb.reserve()
        out1.store(final_cat1)
        out1.push()
        out2 = out2_dfb.reserve()
        out2.store(final_cat2)
        out2.push()
        final_cat2.pop()
        final_cat1.pop()
        final_cat0.pop()
        half_blk.pop()
        one_blk.pop()

    @ttl.datamovement()
    def read():
        spin = spin_dfb.reserve()
        ttl.copy(seed[0, 0], spin).wait()
        spin.push()
        cat0 = cat0_dfb.reserve()
        ttl.copy(seed[1, 0], cat0).wait()
        cat0.push()
        cat1 = cat1_dfb.reserve()
        ttl.copy(seed[2, 0], cat1).wait()
        cat1.push()
        cat2 = cat2_dfb.reserve()
        ttl.copy(seed[3, 0], cat2).wait()
        cat2.push()
        one_blk = one_dfb.reserve()
        ttl.copy(one[0, 0], one_blk).wait()
        one_blk.push()
        half_blk = half_dfb.reserve()
        ttl.copy(half[0, 0], half_blk).wait()
        half_blk.push()
        for step in range(STEPS):
            for category in range(N_CATEGORIES):
                gum = gumbel_dfb.reserve()
                ttl.copy(gumbels[step * N_CATEGORIES + category, 0], gum).wait()
                gum.push()

    @ttl.datamovement()
    def write():
        out0 = out0_dfb.wait()
        ttl.copy(out0, out[0, 0]).wait()
        out0.pop()
        out1 = out1_dfb.wait()
        ttl.copy(out1, out[1, 0]).wait()
        out1.pop()
        out2 = out2_dfb.wait()
        ttl.copy(out2, out[2, 0]).wait()
        out2.pop()


def make_planes(values: list[float]) -> torch.Tensor:
    return torch.stack([torch.full((TILE, TILE), value, dtype=torch.bfloat16) for value in values]).reshape(
        len(values) * TILE,
        TILE,
    )


def make_gumbels() -> tuple[torch.Tensor, list[float]]:
    values = []
    categories = [0.0, 1.0, 0.0]
    for step in range(STEPS):
        step_gumbels = [float(step % 3), float((step + 1) % 3), float((step + 2) % 3)]
        values.extend(step_gumbels)
        scores = [step_gumbels[0] + 1.0, step_gumbels[1] - 1.0, step_gumbels[2]]
        category = max(range(N_CATEGORIES), key=lambda idx: scores[idx])
        categories = [1.0 if idx == category else 0.0 for idx in range(N_CATEGORIES)]
    return make_planes(values), categories


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        seed = to_tt(make_planes([1.0, 0.0, 1.0, 0.0]), device)
        gumbels, expected_categories = make_gumbels()
        one = to_tt(torch.ones((TILE, TILE), dtype=torch.bfloat16), device)
        half = to_tt(torch.full((TILE, TILE), 0.5, dtype=torch.bfloat16), device)
        out = to_tt(torch.zeros((N_CATEGORIES * TILE, TILE), dtype=torch.bfloat16), device)

        categorical_recurrent(seed, to_tt(gumbels, device), one, half, out)

        result = ttnn.to_torch(out).to(torch.float32)
        actual = [float(result[category * TILE, 0]) for category in range(N_CATEGORIES)]
        print("categorical result", actual)
        print("categorical expected", expected_categories)
        assert actual == expected_categories
        print("PASS: categorical recurrent")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
