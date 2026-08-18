# Activation Recomputation

Trading compute for capacity. Symbols: router →
`distributed-training-router/references/notation-and-glossary.md`.

## The trade

Instead of saving every intermediate tensor for backward, save only a few
**checkpoints** and recompute the rest during backward from the nearest
checkpoint.

```
memory:  from  O(L · s·b·h·(34 + 5as/h))   to  O(L · s·b·h · 2)   (layer inputs only)
compute: forward is executed twice for the recomputed regions
```

The FLOP accounting, using `6N` from
`training-metrics/references/flop-counting-and-mfu.md`:

```
without recompute:  2N (fwd) + 4N (bwd)        = 6N per token
with full recompute: 2N (fwd) + 2N (re-fwd) + 4N (bwd) = 8N per token
```

```
FLOP overhead = 8N/6N = 1.333   →  +33%
```

And correspondingly, `HFU/MFU ≈ 8/6` under full recompute. In wall-clock the
overhead is usually **less** than 33% — typically 25–33% — because the
recomputed forward is a dense, compute-bound, cache-friendly pass with no
gradient bookkeeping, and on pipeline schedules part of it lands in what would
otherwise be bubble time.

## Granularity

| Granularity | Memory kept | Overhead | When |
|---|---|---|---|
| None | everything | 0% | if it fits |
| **Selective** (attention only) | everything except the `5as/h` term | **~2–5%** | almost always the right first step |
| Per-layer (checkpoint every layer) | layer inputs, `2·s·b·h·L` | ~25–33% | when selective is not enough |
| Every `k`-th layer | `L/k` checkpoints + one segment recomputed | ~25–33%/`k`-ish | tunable middle ground |
| Full (sqrt-L checkpointing) | `O(√L)` | ~30% | classic result; per-layer is simpler and near-optimal for transformers |

## Selective recomputation is the important one

Look at where the memory actually is. Per layer:

```
s·b·h·(34 + 5as/h)
        ↑     ↑
      34 units  the attention score path
```

At `h = 8192`, `a = 64`, `s = 8192`: `5as/h = 320` against `34`. **The
attention path is 90% of the activation memory** and only a small fraction of
the FLOPs (attention scores are `2sh` FLOPs/token against `12h²` for the
parameterized ops — `s/(6h)` of the total, ~17% here).

So recomputing *only* the attention path removes 90% of the memory for ~2% of
the compute. That is a vastly better trade than uniform per-layer
checkpointing, and it is the headline result of arXiv:2205.05198: **5×
activation-memory reduction while cutting recomputation's execution-time
overhead by over 90%** relative to full recomputation.

```
with selective recompute (+ TP + SP):   34·s·b·h/t   per layer
```

The formula loses the entire `5as/h` term.

**Modern caveat.** A fused attention kernel (FlashAttention) *already*
recomputes the score matrix in its backward pass and never stores it —
`gpu-architecture/references/kernel-fusion-and-flash-attention.md`. If you are
using one, the `5as/h` term is largely gone already, and enabling "selective
recomputation" on top of it buys little. Check which you have before enabling
both.

## What to recompute, in order of value

Rank candidates by memory saved per FLOP recomputed:

| Region | Memory (units) | Recompute cost | Value |
|---|---|---|---|
| attention scores + softmax + dropout | `5as/h` (dominant) | ~`s/(6h)` of FLOPs | **highest** |
| activation function output (`4h` wide) | 8 | one elementwise pass, ~0 FLOPs | **high** — and usually free via fusion |
| layernorm inputs | 4 | trivial | high |
| MLP first-linear output | 8 | a full GEMM | low — expensive to recompute |
| block inputs | 2 | must be kept — they are the checkpoints | n/a |

The pattern: **recompute the cheap, large things.** Elementwise and softmax
outputs are large and nearly free to regenerate; GEMM outputs are the opposite.
This is also why kernel fusion and recomputation overlap so much — a fused
kernel is recomputation done inside one kernel launch.

## Interaction with pipeline parallelism

Under 1F1B, a stage holds up to `p` micro-batches of activations in flight
(`parallelism-strategies/references/pipeline-parallel.md`). Recomputation
therefore has `p`× the value under PP that it has under DP alone — it is
multiplied by the number of in-flight micro-batches.

And some of the recompute cost is absorbed: on schedules with idle time, the
recomputed forward can run in what would otherwise be bubble. This is one
reason large PP runs report recomputation overheads at the low end of the
25–33% band.

## Interaction with ZeRO-3 / FSDP

Recomputation during backward requires the layer's **parameters**, which under
ZeRO-3 are not resident — so the recomputed forward may trigger an *additional*
parameter all-gather. Frameworks handle this by ordering the recompute inside
the backward window where the parameters are already gathered; if that goes
wrong the symptom is a surprising increase in communication volume when
recomputation is enabled. Check comm volume, not just step time, when turning
it on under FSDP.

## Interaction with dropout and RNG

The recomputed forward must produce the **same** dropout masks as the original,
or the gradients are wrong. Frameworks save and restore the RNG state around
the checkpointed region. Two consequences:

- A custom checkpoint implementation that forgets RNG state produces a run
  that trains — badly — with no error.
- Any nondeterministic op inside a checkpointed region is a correctness bug,
  not just a reproducibility annoyance.

## Deciding whether to enable it

```
1. Does it fit without recomputation?        → don't enable it.
2. Are you using a fused attention kernel?   → the 5as/h term is already gone.
3. Still short?  → selective recomputation.  ~2-5%. Almost always sufficient.
4. Still short?  → per-layer, or every k-th layer. ~25-33%.
5. Still short?  → this is a parallelism problem, not a recomputation problem.
                   → distributed-train:parallelism-strategies
```

Step 5 matters. Recomputation's floor is the layer inputs, `2·s·b·h·L`; below
that no amount of recomputation helps and you need to shard.

## Reporting it honestly

Recomputation changes HFU but not MFU. When comparing runs:

- Report **MFU** for cross-config comparison — it is invariant to this choice.
- Report **HFU** alongside if you want to show the hardware was busy.
- Never compare an MFU computed with the `72` coefficient against one computed
  with `96` (`training-metrics/references/flop-counting-and-mfu.md`).

## Sources

- Reducing Activation Recomputation in Large Transformer Models — arXiv:2205.05198
- Efficient Large-Scale LM Training on GPU Clusters — arXiv:2104.04473
- FlashAttention (in-kernel recomputation) — arXiv:2205.14135
