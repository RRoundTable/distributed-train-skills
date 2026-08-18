# Compression and Quantized Collectives

Reducing the *bytes* rather than the number of messages. Symbols: router →
`distributed-training-router/references/notation-and-glossary.md`. Cost model: `T = α + β·S`.

## When compression can possibly help

Compression attacks `β·S` only. It does nothing for `α`, and it *adds* compute
on both ends. So it is worth considering only when all three hold:

1. You are **bandwidth-bound**, not latency-bound (`S ≫ α/β`) and not
   overlap-broken.
2. Communication is **not already hidden** under compute. If overlap works,
   compressing a hidden collective buys nothing.
3. The compression/decompression cost is small relative to the transfer it
   saves — including its effect on the critical path.

The order of operations matters: fix overlap first
(`references/overlap-and-scheduling.md`), fix the parallelism ordering second
(`parallelism-strategies/references/composing-nd-parallelism.md`), and only
then compress. Most "we need gradient compression" conclusions dissolve at
step one or two.

## The precision baseline is not compression

Before anything exotic: gradients in bf16 rather than fp32 halve the wire
bytes at essentially no convergence cost, because the reduction can still
accumulate in fp32 inside the collective. Almost every modern stack does this
by default. If a job is still reducing fp32 gradients, that 2× is the first
and safest win, and it is not "compression" in the risky sense.

## What breaks when you compress gradients

Gradient descent is robust to *unbiased* noise and fragile to *bias*.
Compression schemes are almost all biased — top-k drops the small
coordinates, sign-based methods discard magnitude — so applied naively they
change the optimization problem.

**Error feedback** is the standard repair. Keep the residual locally and add
it back next step:

```
e ← 0
each step:
    g̃    = compress(g + e)      # what actually goes on the wire
    e    = (g + e) − decompress(g̃)   # the part that did not
    apply g̃
```

The dropped signal is not lost, only delayed. With error feedback, biased
compressors converge at rates comparable to uncompressed SGD in theory and
usually in practice. Without it, they can stall or converge to a different
point. **Any compression proposal without error feedback should be treated as
unsound.**

## Families

| Family | Ratio | Mechanism | Main risk |
|---|---|---|---|
| bf16/fp16 gradients | 2× vs fp32 | precision | none in practice |
| fp8 / int8 collectives | 4× vs fp32 | quantize per-block with scales | scale handling; outliers |
| Top-k sparsification | 100–1000× | send only the largest coordinates | needs error feedback; indices cost bytes too; irregular = slow |
| Sign-based (signSGD, 1-bit) | 32× | send only the sign, with a per-block scale | needs error feedback + a warmup phase |
| Low-rank (PowerSGD) | 10–100× | factor the gradient matrix `G ≈ PQᵀ` | rank choice; works best with error feedback and momentum correction |
| Local SGD / infrequent sync | `1/k` communications | sync every `k` steps | model divergence between syncs |

## 1-bit Adam and 1-bit LAMB

The most-deployed instance of this idea in LLM training. The observation is
that Adam's second-moment estimate `v` **stabilizes** after an initial phase.
Once it does, the adaptive scaling is nearly constant, and the remaining
update is essentially momentum SGD — which compresses to 1 bit well with error
feedback.

The scheme is therefore two-phase: a **warmup** phase with uncompressed
communication while `v` settles, then a **compressed** phase using the frozen
`v` and 1-bit momentum. 1-bit LAMB (arXiv:2104.06069) extends this to LAMB's
layerwise adaptive scaling, which is what large-batch training needs, and
reports LAMB's convergence with a large reduction in communication volume.

The practical caveats are real: the warmup length is a hyperparameter, the
frozen `v` is an approximation that can degrade on long runs, and the
implementation must handle the compressed all-reduce itself (a bitwise
all-reduce is not a standard NCCL collective — it is composed from
all-to-all/all-gather primitives, which changes the topology sensitivity).

## Quantized collectives (fp8 / int8 on the wire)

The lower-risk modern option: keep the algorithm, quantize the *transport*.
Split the tensor into blocks, compute a scale per block, send int8/fp8 values
plus the scales, dequantize and reduce in higher precision.

Why it is safer than sparsification: it is (near-)unbiased with stochastic
rounding, the error is bounded relative to the block magnitude, and no
indices are needed so the traffic pattern stays dense and regular — which
means it keeps ring/tree bandwidth optimality.

Why it is not free: reductions must not accumulate in the low precision, so
the collective needs a quantize→transfer→dequantize→reduce structure that not
every library implements efficiently; and per-block scales add overhead that
erodes the ratio for small blocks.

## Sparsification's hidden costs

Top-k at 0.1% density sounds like 1000×, but:

- **Indices are data.** A 32-bit index per surviving value cuts the ratio to
  ~2× that of sending fp32 values densely at that density — the real ratio is
  closer to `1/(2·density)` unless indices are compressed too.
- **Sparse all-reduce is not a thing.** The union of `n` ranks' top-k sets
  grows with `n`, so after reduction the result is much denser than any
  individual contribution. Implementations fall back to all-gathering the
  sparse sets, whose cost grows with `n` — destroying the ring's `2(n−1)S/n`
  optimality.
- **Irregular memory access** makes both the selection and the application
  slow relative to a dense kernel.

Top-k is therefore most defensible at small `n` on genuinely slow links, and
least defensible at the scale where people usually reach for it.

## Decision procedure

```
Is T_comm > T_compute after fixing overlap?
    no  → do not compress. The traffic is hidden.
    yes ↓
Is the collective latency-bound (S < α/β)?
    yes → compression cannot help. Fix message count/bucketing.
    no  ↓
Are gradients still fp32 on the wire?
    yes → switch to bf16. Take the free 2× and re-measure.
    no  ↓
Can the parallelism ordering move this traffic to a faster link?
    yes → do that instead. Zero convergence risk.
    no  ↓
Now consider, in increasing order of risk:
    1. fp8/int8 quantized collectives (unbiased, dense, keeps ring optimality)
    2. 1-bit Adam / LAMB (well-studied, needs warmup tuning)
    3. PowerSGD (needs rank tuning and error feedback)
    4. top-k sparsification (worst scaling behaviour with n)
Always with error feedback. Always validate against an uncompressed baseline
on the same tokens.
```

## Validating a compression change

Compression is a change to the optimization, so it needs the same evidence a
hyperparameter change would:

1. Run the uncompressed baseline and the compressed run on **identical data
   order** for enough steps to see the loss curves separate or not.
2. Compare loss at equal **tokens**, not equal wall-clock — otherwise a faster
   but worse run looks good early.
3. Watch gradient norm, not just loss: compression pathologies usually show up
   in the norm first.
4. Extrapolate cautiously. Compression artifacts that are invisible for 10k
   steps can compound over 100k.

## Sources

- 1-bit LAMB: Communication Efficient Large-Scale Large-Batch Training with
  LAMB's Convergence Speed — arXiv:2104.06069
- ZeRO (for the communication-volume baseline these schemes are measured
  against) — arXiv:1910.02054
