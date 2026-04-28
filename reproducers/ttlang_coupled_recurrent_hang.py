"""THRML-shaped TT-Lang recurrent spin/category hardware hang.

Observed with TT-Lang 1.0.0 on Wormhole:

- simulator: passes
- hardware: hangs during execution

Run from the TT-Lang tooling environment:

    /Users/ludwig/.claude/commands/tools/run-test.sh /abs/path/to/this/file.py
    /Users/ludwig/.claude/commands/tools/run-test.sh --hw /abs/path/to/this/file.py

This is the current whole-window `tt-thrml` blocker. It keeps the carried state
schedule deliberately sequential: each old category front is consumed before the
next accumulator front is pushed, the next spin is held in a separate temporary
CB until the old spin has been consumed, and only then is the spin carry updated.
That shape avoids the obvious simultaneous-carry pattern, but hardware still
hangs while the simulator completes.
"""

from __future__ import annotations

import torch
import ttl
import ttnn

TILE = 32
STEPS = 4
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
def coupled_sequential_window(seed, randomness, one, half, out):
    spin_dfb = ttl.make_dataflow_buffer_like(seed, shape=(1, 1), block_count=2)
    cat0_dfb = ttl.make_dataflow_buffer_like(seed, shape=(1, 1), block_count=2)
    cat1_dfb = ttl.make_dataflow_buffer_like(seed, shape=(1, 1), block_count=2)
    cat2_dfb = ttl.make_dataflow_buffer_like(seed, shape=(1, 1), block_count=2)
    threshold_dfb = ttl.make_dataflow_buffer_like(randomness, shape=(1, 1), block_count=2)
    gumbel_dfb = ttl.make_dataflow_buffer_like(randomness, shape=(1, 1), block_count=2)
    one_dfb = ttl.make_dataflow_buffer_like(one, shape=(1, 1), block_count=2)
    half_dfb = ttl.make_dataflow_buffer_like(half, shape=(1, 1), block_count=2)
    spin_accum_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    next_spin_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    score0_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    score1_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    score2_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    spin_out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    cat0_out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    cat1_out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    cat2_out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        one_blk = one_dfb.wait()
        half_blk = half_dfb.wait()

        for _ in range(STEPS):
            threshold = threshold_dfb.wait()
            zero = spin_accum_dfb.reserve()
            zero.store(threshold - threshold)
            zero.push()

            accum0 = spin_accum_dfb.wait()
            cat0 = cat0_dfb.wait()
            accum1 = spin_accum_dfb.reserve()
            accum1.store(accum0 + cat0)
            accum1.push()
            cat0.pop()
            accum0.pop()

            accum1 = spin_accum_dfb.wait()
            cat1 = cat1_dfb.wait()
            accum2 = spin_accum_dfb.reserve()
            accum2.store(accum1 + cat1)
            accum2.push()
            cat1.pop()
            accum1.pop()

            accum2 = spin_accum_dfb.wait()
            cat2 = cat2_dfb.wait()
            accum3 = spin_accum_dfb.reserve()
            accum3.store(accum2 + cat2)
            accum3.push()
            cat2.pop()
            accum2.pop()

            accum3 = spin_accum_dfb.wait()
            next_spin_tmp = next_spin_dfb.reserve()
            decision = ttl.math.sign(accum3 + accum3 - threshold)
            next_spin_tmp.store(ttl.math.sign(decision - half_blk))
            next_spin_tmp.push()
            accum3.pop()
            threshold.pop()

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
            next_cat0 = cat0_dfb.reserve()
            next_cat0.store(gt01 * gt02)
            next_cat0.push()

            gt10 = (ttl.math.sign(score1_blk - score0_blk) + one_blk) * half_blk
            gt12 = (ttl.math.sign(score1_blk - score2_blk) + one_blk) * half_blk
            next_cat1 = cat1_dfb.reserve()
            next_cat1.store(gt10 * gt12)
            next_cat1.push()

            gt20 = (ttl.math.sign(score2_blk - score0_blk) + one_blk) * half_blk
            gt21 = (ttl.math.sign(score2_blk - score1_blk) + one_blk) * half_blk
            next_cat2 = cat2_dfb.reserve()
            next_cat2.store(gt20 * gt21)
            next_cat2.push()
            score2_blk.pop()
            score1_blk.pop()
            score0_blk.pop()
            old_spin.pop()

            next_spin_tmp = next_spin_dfb.wait()
            next_spin = spin_dfb.reserve()
            next_spin.store(next_spin_tmp)
            next_spin.push()
            next_spin_tmp.pop()

        final_spin = spin_dfb.wait()
        out_spin = spin_out_dfb.reserve()
        out_spin.store(final_spin)
        out_spin.push()
        final_spin.pop()
        final_cat0 = cat0_dfb.wait()
        out_cat0 = cat0_out_dfb.reserve()
        out_cat0.store(final_cat0)
        out_cat0.push()
        final_cat0.pop()
        final_cat1 = cat1_dfb.wait()
        out_cat1 = cat1_out_dfb.reserve()
        out_cat1.store(final_cat1)
        out_cat1.push()
        final_cat1.pop()
        final_cat2 = cat2_dfb.wait()
        out_cat2 = cat2_out_dfb.reserve()
        out_cat2.store(final_cat2)
        out_cat2.push()
        final_cat2.pop()
        half_blk.pop()
        one_blk.pop()

    @ttl.datamovement()
    def read():
        spin_seed = spin_dfb.reserve()
        ttl.copy(seed[0, 0], spin_seed).wait()
        spin_seed.push()
        cat0_seed = cat0_dfb.reserve()
        ttl.copy(seed[1, 0], cat0_seed).wait()
        cat0_seed.push()
        cat1_seed = cat1_dfb.reserve()
        ttl.copy(seed[2, 0], cat1_seed).wait()
        cat1_seed.push()
        cat2_seed = cat2_dfb.reserve()
        ttl.copy(seed[3, 0], cat2_seed).wait()
        cat2_seed.push()
        one_blk = one_dfb.reserve()
        ttl.copy(one[0, 0], one_blk).wait()
        one_blk.push()
        half_blk = half_dfb.reserve()
        ttl.copy(half[0, 0], half_blk).wait()
        half_blk.push()
        for step in range(STEPS):
            threshold = threshold_dfb.reserve()
            ttl.copy(randomness[step * LANES, 0], threshold).wait()
            threshold.push()
            for category in range(3):
                gum = gumbel_dfb.reserve()
                ttl.copy(randomness[step * LANES + 1 + category, 0], gum).wait()
                gum.push()

    @ttl.datamovement()
    def write():
        spin = spin_out_dfb.wait()
        ttl.copy(spin, out[0, 0]).wait()
        spin.pop()
        cat0 = cat0_out_dfb.wait()
        ttl.copy(cat0, out[1, 0]).wait()
        cat0.pop()
        cat1 = cat1_out_dfb.wait()
        ttl.copy(cat1, out[2, 0]).wait()
        cat1.pop()
        cat2 = cat2_out_dfb.wait()
        ttl.copy(cat2, out[3, 0]).wait()
        cat2.pop()


def make_planes(values: list[float]) -> torch.Tensor:
    return torch.stack([torch.full((TILE, TILE), value, dtype=torch.bfloat16) for value in values]).reshape(
        len(values) * TILE,
        TILE,
    )


def make_randomness() -> tuple[torch.Tensor, list[float]]:
    random_planes = []
    spin = 1.0
    categories = [0.0, 1.0, 0.0]
    for step in range(STEPS):
        threshold = 0.5 if step % 2 == 0 else 3.0
        gumbels = [float(step % 3), float((step + 1) % 3), float((step + 2) % 3)]
        random_planes.append(torch.full((TILE, TILE), threshold, dtype=torch.bfloat16))
        random_planes.extend(torch.full((TILE, TILE), value, dtype=torch.bfloat16) for value in gumbels)
        old_spin = spin
        spin = 1.0 if (sum(categories) * 2.0) > threshold else -1.0
        scores = [gumbels[0] + old_spin, gumbels[1] - old_spin, gumbels[2]]
        category = max(range(3), key=lambda idx: scores[idx])
        categories = [1.0 if idx == category else 0.0 for idx in range(3)]
    return torch.stack(random_planes).reshape(STEPS * LANES * TILE, TILE), [spin, *categories]


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        seed = to_tt(make_planes([1.0, 0.0, 1.0, 0.0]), device)
        randomness_torch, expected_lanes = make_randomness()
        randomness = to_tt(randomness_torch, device)
        one = to_tt(torch.ones((TILE, TILE), dtype=torch.bfloat16), device)
        half = to_tt(torch.full((TILE, TILE), 0.5, dtype=torch.bfloat16), device)
        out = to_tt(torch.zeros((LANES * TILE, TILE), dtype=torch.bfloat16), device)

        coupled_sequential_window(seed, randomness, one, half, out)

        result = ttnn.to_torch(out).to(torch.float32)
        expected = torch.stack(
            [torch.full((TILE, TILE), value, dtype=torch.float32) for value in expected_lanes]
        ).reshape(LANES * TILE, TILE)
        print("result", [float(result[lane * TILE, 0]) for lane in range(LANES)])
        print("expected", [float(expected[lane * TILE, 0]) for lane in range(LANES)])
        assert torch.allclose(result, expected, rtol=0, atol=0), "coupled sequential mismatch"
        print("PASS: coupled sequential window")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
