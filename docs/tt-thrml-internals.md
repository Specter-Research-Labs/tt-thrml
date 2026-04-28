# tt-thrml Internals

`tt-thrml` lowers supported THRML programs directly into TT-Lang-oriented
state and runtime kernels.

## State Layout

THRML host state keeps the original shapes:

- spin blocks: booleans
- categorical blocks: scalar category ids
- gaussian blocks: floats

The runtime state is a single TT-Lang lane tensor:

- spin blocks become signed `-1/+1` lanes
- categorical blocks become one-hot category lanes
- gaussian blocks become value lanes

The current mixed smoke program has six THRML blocks and ten TT-Lang lanes:

```text
block 0 spin        lane 0
block 1 categorical lanes 1..3
block 2 gaussian    lane 4
block 3 spin        lane 5
block 4 categorical lanes 6..8
block 5 gaussian    lane 9
```

## Planning

`TTLangProgramPlanner` reads the THRML `BlockSamplingProgram`, infers each block
family, folds active masks into interaction weights, translates THRML gather
slices into flat scalar indices, and then expands categorical scalar gathers
into one-hot lane gathers.

The current hardware-proven planner accepts:

- one-node spin targets with categorical sources
- one-node categorical targets with spin sources
- two Gibbs groups, each containing one spin update and one categorical update

Unsupported shapes raise during planning or runtime validation.

## Runtime

`TTLangDiscreteSweepRuntime` owns device-resident state buffers and TT-Lang
operations. One sweep currently runs:

1. copy pre-group state to the group output buffer
2. update the group's spin block
3. update the group's categorical block
4. repeat for the second group

That is six TT-Lang dispatches per sweep. The next major optimization target is
to reduce dispatch count by fusing per-group copy and update work without
breaking TT-Lang dataflow balance.

Hardware note: bad fused-kernel experiments can leave Wormhole dispatch cores
running after the host process is killed. If TT-Metal reports unexpected
`run_mailbox` values or active ethernet dispatch cores during initialization,
clear the board with `tt-smi -r all --no_reinit` from the TT-Lang container
before rerunning benchmarks.

## Randomness

The current runner exposes deterministic per-block inputs:

- spin threshold logits
- categorical Gumbel values

Those constants are uploaded to device tensors before sweeps. They can also be
derived from a JAX key with `TTLangProgramPlanner.sweep_randomness_from_key` or
`TTLangDiscreteSweepRuntime.set_sweep_randomness_from_key`.

The key derivation matches THRML's one-sweep sampling path:

1. split the sweep key once per free block
2. split each block key once inside the parametric sampler
3. use the first sampler subkey for the Bernoulli or categorical draw

For TT-Lang, Bernoulli draws are represented as logit thresholds and categorical
draws are represented as Gumbel-max perturbations. The next runtime step is to
batch full sweep windows in device-resident randomness buffers rather than
replacing constants one sweep at a time.

## Device Ownership

The caller opens TTNN devices and passes them to `tt_thrml.make_executor`.
Executors borrow devices and never close them.
