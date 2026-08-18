# Data Parallelism and ZeRO

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`. This file is where
the `2Ψ + 2Ψ + KΨ` model-state algebra is **derived**; other files cite it.

## DDP: replicate everything, reduce the gradients

Each of `d` ranks holds a full copy of the model, processes a different slice
of the batch, and all-reduces gradients before the optimizer step. Because the
loss is a mean over samples and the gradient of a mean is the mean of the
gradients, averaging gradients across ranks is *exactly* equivalent to one
large batch — provided every rank contributed the same number of tokens.
When they did not, the mean-of-means is wrong; see
`training-metrics/references/loss-curve-diagnostics.md`.

Per step, per rank: one all-reduce over `Ψ` gradient elements. In busbw terms
that is `2(n−1)/n · Ψ · bytes` ≈ `2Ψ` elements of traffic.

The critical property is **overlap**: gradients for layer `L` are ready while
layer `L−1` is still computing its backward. DDP buckets gradients and fires
the all-reduce for a bucket as soon as it fills, so most of the `2Ψ` hides
under backward compute (arXiv:2006.15704). Bucket sizing is derived in
`communication-backends/references/overlap-and-scheduling.md`.

DDP's problem is not speed. It is that **capacity does not scale**: every
rank stores the full model state regardless of `d`.

## The model-state budget: `2Ψ + 2Ψ + KΨ`

Mixed-precision training with Adam stores, per parameter:

| Component | dtype | bytes/param | Total |
|---|---|---|---|
| compute weights | bf16 | 2 | `2Ψ` |
| gradients | bf16 | 2 | `2Ψ` |
| fp32 master weights | fp32 | 4 | part of `KΨ` |
| Adam first moment `m` | fp32 | 4 | part of `KΨ` |
| Adam second moment `v` | fp32 | 4 | part of `KΨ` |

So `K = 12` for fp32-master Adam, and the total is:

```
model_states = 2Ψ + 2Ψ + KΨ = 16Ψ bytes        (K = 12)
```

For a 7B model: `16 × 7e9 = 112 GB`. It does not fit on an 80GB card, and
that is before a single activation. This one number is why ZeRO exists.

`K` varies with the optimizer and precision choice:

| Setup | `K` | Total bytes/param |
|---|---|---|
| Adam, fp32 master + fp32 moments | 12 | 16 |
| Adam, bf16 moments (no master) | 4 | 8 |
| SGD + momentum, fp32 master | 8 | 12 |
| Adafactor (factored second moment) | ≈4 + `O(h)` | ≈8 |
| 8-bit Adam | 2 + 4 (master) | 10 |

`memory-offloading/references/optimizer-and-precision-memory.md` covers the
tradeoffs of each and why the fp32 master copy is usually not optional.

## ZeRO: shard what is replicated

ZeRO (arXiv:1910.02054) observes that all three components are *identical*
across the `d` data-parallel ranks and therefore pure waste. Shard them.

| Stage | Sharded | Per-rank model-state bytes | Comm volume |
|---|---|---|---|
| DDP | nothing | `16Ψ` | `2Ψ` |
| ZeRO-1 | optimizer state | `2Ψ + 2Ψ + KΨ/d` | `2Ψ` |
| ZeRO-2 | + gradients | `2Ψ + (2Ψ + KΨ)/d` | `2Ψ` |
| ZeRO-3 | + parameters | `(2Ψ + 2Ψ + KΨ)/d = 16Ψ/d` | `3Ψ` |

For 7B on `d = 8`:

| Stage | GB per rank |
|---|---|
| DDP | 112 |
| ZeRO-1 | 28 + 10.5 = 38.5 |
| ZeRO-2 | 14 + 12.25 = 26.25 |
| ZeRO-3 | 14 |

## Why ZeRO-1 and ZeRO-2 are free

An all-reduce decomposes exactly:

```
AllReduce = ReduceScatter ∘ AllGather
```

Ring all-reduce already *implements* it this way — see
`communication-backends/references/collective-algorithms.md`. So under DDP
each rank is already producing a fully-reduced shard partway through the
collective, then gathering it back.

ZeRO-1 exploits this. Instead of `AllReduce(grads)` then a full optimizer
step, it does:

1. `ReduceScatter(grads)` — each rank ends with the reduced gradient for
   *its* shard only. Cost: `Ψ` elements.
2. Each rank runs the optimizer on its `1/d` of the state, updating its `1/d`
   of the parameters. Cost: none on the wire, and `1/d` the optimizer FLOPs.
3. `AllGather(params)` — everyone gets the updated full parameters. Cost:
   `Ψ` elements.

Total: `2Ψ`. Identical to DDP. ZeRO-2 additionally frees the gradient buffer
after the reduce-scatter, which costs nothing extra because the gradient for
a shard is only needed by its owner.

**Practical reading: if you run DDP and are anywhere near a memory limit,
ZeRO-2 is strictly better. There is no bandwidth argument for DDP over
ZeRO-2.** The remaining arguments for DDP are implementation simplicity and
the fact that ZeRO-2's reduce-scatter overlaps slightly less cleanly with
backward than DDP's bucketed all-reduce in some frameworks.

## Why ZeRO-3 costs 1.5×

Under ZeRO-3 no rank holds a full layer's parameters at rest. To compute
layer `i`, its parameters must be all-gathered *just in time*, used, then
freed. And the backward pass needs them again:

```
forward:   AllGather(params)              →  Ψ
backward:  AllGather(params) again        →  Ψ
backward:  ReduceScatter(grads)           →  Ψ
                                     total = 3Ψ
```

versus `2Ψ` for ZeRO-1/2 — the **1.5× rule**. In exchange, per-rank model
state drops to `16Ψ/d`, the largest capacity win available without touching
the model definition.

The second all-gather can be avoided by *keeping* the forward-gathered
parameters until backward, but that reinstates full parameter residency and
defeats the point. Frameworks expose this as a per-layer choice
(`reshard_after_forward` in FSDP): keeping the outermost layers gathered is a
cheap partial recovery, since the first backward to need them runs soonest.

## FSDP and DeepSpeed ZeRO-3: same semantics, different shape

| | PyTorch FSDP | DeepSpeed ZeRO-3 |
|---|---|---|
| Sharding unit | an `FSDPModule` / wrapped submodule | the flat parameter partition |
| Prefetch | explicit forward/backward prefetch of the next unit | automatic, param-coordinator driven |
| Reshard-after-forward | per-module toggle | `stage3_max_live_parameters` and friends |
| Mixed precision | `MixedPrecisionPolicy` per module | ds_config `bf16`/`fp16` block |
| Checkpointing | `DCP` sharded state dict | ZeRO checkpoint, needs conversion for HF |
| CPU/NVMe offload | CPU offload of params/grads | full ZeRO-Offload / Infinity stack |

The single most consequential FSDP knob is **wrapping granularity**. Wrap too
coarsely and the all-gather for a huge unit cannot overlap with anything;
wrap too finely and you pay `α` latency per tiny collective. The right unit is
usually one transformer block: large enough to amortize latency, small enough
that the gather for block `i+1` overlaps the compute of block `i`.
(arXiv:2304.11277 for FSDP's design; the latency-vs-bandwidth tradeoff is the
same `k(α+βs)` vs `α+βks` algebra as gradient bucketing.)

## Gradient accumulation interacts with all of this

With `m` accumulation steps, ranks run `m` forward/backwards before
communicating. Under DDP you must *suppress* the all-reduce on the first
`m−1` micro-batches (`no_sync()`); otherwise you pay `m ×` the communication
for the same step. Under ZeRO-3, parameters must still be gathered every
micro-batch — accumulation reduces the *gradient* traffic but not the
*parameter* traffic, so ZeRO-3's advantage from accumulation is smaller.

This is the cleanest lever on the compute-to-communication ratio: `m` appears
in the numerator of `6·b·m·s / bytes_per_elem` (router →
`distributed-training-router/references/why-scaling-is-hard.md`) and, under DDP, not in the denominator
at all.

## What ZeRO does not fix

ZeRO shards **model states**. It does nothing for **activations**, which
scale as `s·b·h·L` and dominate at long sequence — that is TP's, CP's, and
recomputation's job. A ZeRO-3 job can still OOM in the middle of a forward
pass with almost no parameter memory resident. The activation formula is in
`memory-offloading/references/memory-taxonomy-and-math.md`.

## Sources

- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models — arXiv:1910.02054
- PyTorch Distributed: Experiences on Accelerating Data Parallel Training — arXiv:2006.15704
- PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel — arXiv:2304.11277
- ZeRO-Offload — arXiv:2101.06840 · ZeRO-Infinity — arXiv:2104.07857
