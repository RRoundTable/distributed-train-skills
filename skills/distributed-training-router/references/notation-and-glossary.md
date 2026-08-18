# Notation and Glossary

The single symbol table for this plugin. Every other file in
`distributed-train` uses these symbols and does not redefine them. If a
formula elsewhere uses a bare letter, look it up here.

## Symbol table

| Sym | Meaning | Typical value |
|---|---|---|
| `Ψ` | number of model **parameters** (elements, not bytes) | 7e9, 70e9, 405e9 |
| `N` | parameters counted for FLOPs (non-embedding, unless stated) | ≈ `12·L·h²` |
| `n` | total accelerators (world size) | 8 … 16384 |
| `L` | transformer layers | 32 (7B), 126 (405B) |
| `h` | hidden / model dimension | 4096 (7B), 16384 (405B) |
| `a` | attention heads | 32, 128 |
| `s` | sequence length (tokens per sample) | 2048 … 131072 |
| `b` | **micro**-batch size (samples per forward on one rank) | 1 … 8 |
| `m` | gradient-accumulation steps (micro-batches per rank per step) | 1 … 64 |
| `B` | **global** batch size in samples | `b·m·d` |
| `D` | tokens consumed (dataset size, or tokens in a step) | 15e12 |
| `V` | vocabulary size | 32000, 50257, 128256 |
| `d` | data-parallel degree | |
| `t` | tensor-parallel degree | 1, 2, 4, 8 |
| `p` | pipeline-parallel degree | 1 … 16 |
| `c` | context/ring-parallel degree | 1 … 16 |
| `e` | expert-parallel degree (MoE) | |
| `v` | virtual pipeline stages per device (interleaving) | 1 … 8 |
| `K` | optimizer-state bytes per parameter-byte multiplier | 12 for fp32 Adam |
| `α` | per-message latency of a link (seconds) | 1–10 µs |
| `β` | inverse bandwidth, seconds per byte (`1/BW`) | |
| `S` | message size in bytes for one collective | |
| `P_peak` | dense peak FLOP/s of one accelerator at the training dtype | |
| `T` | wall-clock seconds per optimizer step | |

## The two invariants

Everything in this plugin is consistent with exactly two identities. When a
config change breaks a result, check these first.

**Mesh invariant** — every accelerator belongs to exactly one coordinate in
each parallel axis, so the degrees multiply to the world size:

```
n = d · t · p · c        (and n = d · t · p · c · e when experts are sharded)
```

**Batch invariant** — the global batch is the micro-batch times accumulation
times the number of data-parallel replicas:

```
B = b · m · d            tokens per step = B · s
```

`t`, `p`, and `c` do **not** appear in `B`. Ranks inside a tensor, pipeline,
or context group all process *the same* samples; only `d` multiplies data.
This is the single most common source of "my loss changed when I changed the
parallelism config" — see `training-metrics/references/loss-curve-diagnostics.md`.

## Bytes per element

| dtype | bytes | where it shows up |
|---|---|---|
| fp32 | 4 | master weights, Adam moments, loss accumulation |
| tf32 | 4 (stored) | fp32 storage, reduced-mantissa matmul on Ampere+ |
| bf16 | 2 | compute weights, gradients, activations |
| fp16 | 2 | same as bf16 but needs loss scaling |
| fp8 (e4m3 / e5m2) | 1 | Hopper+ matmul inputs, rarely master state |
| int8 | 1 | quantized collectives, inference |

A "13B model in bf16" is `13e9 × 2 = 26 GB` of *weights alone*. Weights are
never the whole story — see `memory-offloading/references/memory-taxonomy-and-math.md`.

## Units hygiene — the two mistakes that cost people hours

**1. Gb/s vs GB/s.** Network vendors quote **giga*bits*** per second; memory
and collective libraries quote **giga*bytes*** per second. Divide by 8.

| Fabric | Vendor figure | Bytes/s |
|---|---|---|
| 100 GbE | 100 Gb/s | 12.5 GB/s |
| HDR InfiniBand | 200 Gb/s | 25 GB/s |
| NDR InfiniBand | 400 Gb/s | 50 GB/s |
| NVLink (H100, per GPU, bidirectional) | 900 GB/s | already bytes |

A "400G" node with 8 GPUs sharing one NIC gives each GPU ~6.25 GB/s of
egress — two orders of magnitude below its own HBM. That ratio is why
parallelism placement matters at all.

**2. `algbw` vs `busbw`.** For a collective moving `S` bytes of *user* data
in time `T`:

```
algbw = S / T                        (what your application sees)
busbw = algbw · f(n)                 (what the wire actually carries)
```

The correction factor `f(n)` differs per collective, because a ring
all-reduce sends each byte around the ring roughly twice:

| Collective | `f(n)` |
|---|---|
| all-reduce | `2(n−1)/n` |
| all-gather, reduce-scatter | `(n−1)/n` |
| broadcast, reduce | `1` |
| all-to-all | `(n−1)/n` |

`busbw` is the number to compare against link speed, because it saturates at
the hardware peak while `algbw` does not. `nccl-tests` prints both; people
compare the wrong column and conclude the fabric is broken.

## Glossary

**Activation** — a tensor produced in the forward pass and kept because the
backward pass needs it. Dominated by, and scaling with, `s·b·h`, not `Ψ`.

**All-gather (AG)** — each rank contributes a shard, every rank ends with the
full tensor. Moves `S(n−1)/n` bytes per rank.

**All-reduce (AR)** — element-wise reduction whose result every rank holds.
Decomposes exactly as `AR = ReduceScatter ∘ AllGather`.

**Arithmetic intensity (AI)** — FLOPs performed per byte moved from HBM. The
x-axis of the roofline. See `gpu-architecture/references/roofline-and-arithmetic-intensity.md`.

**Bubble** — pipeline time in which a stage has no micro-batch to run.
Fraction `(p−1)/(m+p−1)` for the naive schedule.

**Collective** — a communication primitive all ranks in a group call
together. Every collective is also a **barrier**: no rank leaves until all
arrive. This is why a single slow rank stalls the world.

**Context parallelism (CP)** — sharding the sequence dimension *including*
attention, requiring K/V exchange between ranks. Distinct from sequence
parallelism.

**FSDP** — PyTorch's implementation of ZeRO-3 semantics: parameters,
gradients, and optimizer state are all sharded, and parameters are
all-gathered just in time per module.

**HFU / MFU** — hardware vs model FLOPs utilization. HFU counts every FLOP
the chip executed (including recomputation); MFU counts only the FLOPs the
model definition requires. `MFU ≤ HFU` always.

**Micro-batch** — the unit of one forward/backward on one rank (`b`). The
unit of pipeline scheduling and the first knob to turn on OOM.

**Ring** — a communication pattern where rank `i` only ever talks to `i±1`.
Bandwidth-optimal, latency `O(n)`.

**Sequence parallelism (SP)** — Megatron's term: sharding *only* the
layernorm/dropout regions along `s` to shrink activation memory, with no
change to attention. **Not** the same as CP.

**Straggler** — a rank that is systematically slower, turning every
collective into a wait on it. Shows up as step-time variance, not as an error.

**ZeRO stage** — which of {optimizer state, gradients, parameters} is
sharded across the data-parallel group: stage 1 = optimizer, stage 2 = +
gradients, stage 3 = + parameters.

## Where each symbol is derived

| Result | Derived in |
|---|---|
| `2Ψ + 2Ψ + KΨ` model-state bytes, ZeRO stages | `parallelism-strategies/references/data-parallel-and-zero.md` |
| `s·b·h·(34 + 5as/h)` activation bytes | `memory-offloading/references/memory-taxonomy-and-math.md` |
| ring all-reduce `2(n−1)S/n` optimality | `communication-backends/references/collective-algorithms.md` |
| online softmax / FlashAttention | `gpu-architecture/references/kernel-fusion-and-flash-attention.md` |
| `6N` FLOPs per token, `MFU = 6ND/(T·P_peak·n)` | `training-metrics/references/flop-counting-and-mfu.md` |
