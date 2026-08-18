# Pipeline Parallelism

Splitting the *layer stack* across `p` stages. Symbols: router →
`notation-and-glossary.md`.

## The bubble, derived

Stage `i` cannot start micro-batch `j`'s forward until stage `i−1` finished
it. With `m` micro-batches and `p` stages, the pipeline needs `p−1` steps to
fill and `p−1` to drain, during which some stages sit idle.

Count in units of one micro-batch's forward+backward on one stage:

```
useful work per stage      = m
total pipeline length      = m + p − 1
bubble fraction            = (m + p − 1 − m)/(m + p − 1) = (p − 1)/(m + p − 1)
```

```
bubble = (p − 1) / (m + p − 1)
```

| `p` | `m=1` | `m=4` | `m=8` | `m=16` | `m=64` |
|---|---|---|---|---|---|
| 2 | 50% | 20% | 11% | 6% | 1.5% |
| 4 | 75% | 43% | 27% | 16% | 4.5% |
| 8 | 88% | 64% | 47% | 30% | 10% |
| 16 | 94% | 79% | 65% | 48% | 19% |

Two readings:

1. **`m ≫ p` is not advice, it is a requirement.** Below `m ≈ 4p` the bubble
   is double digits.
2. **`m` is not free.** `B = b·m·d`, so at fixed global batch, raising `m`
   forces `d` down. PP and DP compete for the same budget. This is the actual
   reason large runs cap `p` in the 8–16 range rather than pushing further.

### The third lever on `m`: raise `B` itself

The constraint above is `B = b·m·d` at *fixed* `B`. But `B` is a
hyperparameter, not a physical limit — it is fixed only because convergence at
a larger batch is not free. That makes the optimizer a pipeline-efficiency
knob:

```
larger B at fixed d, b   →   larger m   →   smaller bubble (p−1)/(m+p−1)
```

MegaScale took this route explicitly, using LAMB (arXiv:1904.00962) to scale
the batch by 4× without accuracy loss in their setting, and reports **87.5% of
the pipeline bubble removed** under an interleaved schedule as a result
(arXiv:2402.15627). Their ablation prices the whole move at +3.0 MFU points.

This is worth internalizing as a pattern, not just a trick: **an algorithmic
choice bought a systems property.** The bubble is usually treated as fixed by
`p` and `m`, with `m` fixed by `B` — and one of those "fixed" values is a
hyperparameter someone chose.

The cost is real and belongs in the decision:

- past the critical batch size, more samples per step stop buying fewer steps
  (`training-metrics/references/compute-budget-and-scaling-laws.md`), so a
  throughput win can be a time-to-loss loss
- large-batch optimizers change the optimization problem; "no accuracy loss at
  4×" is a result on one model family, not a law
- larger `B` at fixed `d` raises activation memory through `b` or `m`
  (`memory-offloading/references/activation-recomputation.md`)

Verify it as a **time-to-target-loss** win, not a tokens/s win.

## Schedules

### GPipe — all-forward-then-all-backward

Run all `m` forwards, then all `m` backwards. Simple, and the bubble is
`(p−1)/(m+p−1)`. Its problem is memory: every micro-batch's activations for a
stage are alive simultaneously, so activation memory scales with `m`, exactly
the parameter you wanted to raise. (arXiv:1811.06965)

### 1F1B / PipeDream-Flush — one forward, one backward

After the warm-up, each stage alternates: one forward, one backward. The
bubble is **identical** to GPipe, but at most `p` micro-batches' activations
are alive at once instead of `m`. Since `m > p` is exactly the regime you
want, 1F1B is strictly better and is the default in Megatron-LM and DeepSpeed.
(arXiv:1806.03377; the double-buffered-weights variant, arXiv:2006.09503.)

The steady-state activation count is not uniform: stage 0 holds `p`
micro-batches in flight, the last stage holds 1. Memory pressure is therefore
**highest on the first stage** — a fact that drives stage balancing below.

### Interleaved 1F1B — `v` virtual stages per device

Give each device `v` non-contiguous chunks of layers instead of one
contiguous block (device 0 gets layers 0–3 *and* 16–19, etc.). Each chunk is
smaller, so the fill/drain overlap tightens:

```
bubble_interleaved = (1/v) · (p − 1)/m
```

At `p=8`, `m=32`, `v=1` the bubble is 18%; at `v=4` it is 5.5%. The cost is
`v×` as many point-to-point messages — each smaller, so the `α` latency term
grows. Interleaving is a good trade on NVLink-connected stages and a poor one
across a high-latency fabric. (arXiv:2104.04473)

### Zero Bubble

Backward for a layer is really two computations: the gradient with respect to
the *input* (needed immediately by the previous stage) and the gradient with
respect to the *weights* (needed only before the optimizer step). Splitting
them lets weight-gradient work fill what would be bubble. With a carefully
constructed schedule and optimizer-step bypass, the bubble goes to
approximately zero. (arXiv:2401.10241)

### Chimera — bidirectional pipelines

Run two pipelines in opposite directions over the same devices, so each
device is a late stage in one and an early stage in the other. Halves the
bubble at the cost of holding two model replicas' worth of stage weights.
(arXiv:2107.06925)

| Schedule | Bubble | Activation memory | Extra cost |
|---|---|---|---|
| GPipe | `(p−1)/(m+p−1)` | `O(m)` per stage | — |
| 1F1B | `(p−1)/(m+p−1)` | `O(p)` per stage | — |
| Interleaved 1F1B (`v`) | `(1/v)·(p−1)/m` | `O(p)` | `v×` p2p messages |
| Zero Bubble | ≈0 | `O(p)` | schedule complexity, optimizer bypass |
| Chimera | ~half | `O(p)`, 2× stage weights | duplicate weights |

## Communication: small volume, large consequence

Each stage boundary sends the activation tensor forward and the gradient
backward — point-to-point, not a collective:

```
per micro-batch per boundary:  b·s·h forward  +  b·s·h backward  = 2·b·s·h
per step:                       m · 2 · b·s·h  per boundary
```

For `b=1`, `s=8192`, `h=8192`, bf16: 134 MB per micro-batch per boundary.
That is small next to TP's `4·b·s·h` **per layer**. PP is cheap on the wire
and expensive in *serialization* — the bubble, not the bytes, is the cost.
This asymmetry is exactly why PP is the axis assigned to slow inter-node
links.

## Stage balancing

The pipeline runs at the speed of its slowest stage, so equal layer counts
per stage are usually *not* balanced:

- Stage 0 carries the embedding; the last stage carries the output projection
  and loss — both large relative to a transformer block, especially at large
  `V`.
- Under 1F1B, stage 0 holds `p` micro-batches of activations while the last
  holds 1. Memory imbalance, not compute imbalance.

Common fixes: give the first and last stages fewer transformer layers; place
the embedding and output projection on different stages; or use interleaving,
which averages the imbalance across chunks by construction.

## PP and activation recomputation interact

Under 1F1B, a stage holds up to `p` micro-batches of activations. If you also
enable full activation recomputation, only the layer inputs survive — memory
drops sharply, and the *recompute* work happens to land in what would
otherwise be bubble time on some schedules. PP therefore hides part of
recomputation's cost that DP-only training would pay in full. Pricing is in
`memory-offloading/references/activation-recomputation.md`.

## When PP is the right tool

| Situation | PP? |
|---|---|
| Model does not fit on one node even with TP=8 + ZeRO | **yes** — PP is the axis that crosses nodes cheaply |
| Slow inter-node fabric, fast intra-node | **yes** — p2p volume is small |
| `B` too small to give `m ≥ 4p` | **no** — the bubble will dominate |
| Model fits with ZeRO-3 and the fabric is fast | **no** — ZeRO-3 avoids the bubble and the schedule complexity |
| You need elastic/dynamic world size | **no** — PP fixes the stage assignment |

## Diagnosing a bubble in a profile

A pipeline bubble has a distinctive signature: **regular, repeating idle
blocks at a fixed offset within each step**, identical across steps, longer on
the outer stages. Compare against:

- straggler → *irregular* idle, varying which rank is late
  (`communication-backends/references/hang-and-straggler-debugging.md`)
- bandwidth-bound → idle *inside* NCCL kernels, not between kernels
- CPU-bound launch → gaps everywhere, including within a stage

If measured idle exceeds `(p−1)/(m+p−1)` substantially, the stages are
imbalanced rather than merely bubbling.

## Sources

- GPipe — arXiv:1811.06965
- PipeDream — arXiv:1806.03377 · PipeDream-2BW — arXiv:2006.09503
- PTD-P / interleaved schedule — arXiv:2104.04473
- Zero Bubble Pipeline Parallelism — arXiv:2401.10241
- Chimera — arXiv:2107.06925
- LAMB / large-batch optimization — arXiv:1904.00962
- MegaScale (bubble reduction via larger batch, at 10k+ GPUs) — arXiv:2402.15627
