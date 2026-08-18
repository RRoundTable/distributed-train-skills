# Optimizer and Precision Memory

The `KΨ` term. Symbols: router → `distributed-training-router/references/notation-and-glossary.md`.

## `K` by optimizer and precision

`K` is bytes of optimizer state (including the fp32 master copy) per
parameter. Total model states are `(2 + 2 + K)·Ψ`.

| Setup | Master | Moments | `K` | Total bytes/param | 7B total |
|---|---|---|---|---|---|
| Adam/AdamW, fp32 master + fp32 moments | 4 | 4 + 4 | **12** | 16 | 112 GB |
| Adam, fp32 master + bf16 moments | 4 | 2 + 2 | **8** | 12 | 84 GB |
| Adam, bf16 master + bf16 moments | 0 | 2 + 2 | **4** | 8 | 56 GB |
| 8-bit Adam, fp32 master | 4 | 1 + 1 | **6** | 10 | 70 GB |
| SGD + momentum, fp32 master | 4 | 4 | **8** | 12 | 84 GB |
| SGD, no momentum, fp32 master | 4 | 0 | **4** | 8 | 56 GB |
| Adafactor (factored 2nd moment) | 4 | 4 + `O(h)` | **≈8** | ≈12 | ≈84 GB |
| Lion (one momentum buffer) | 4 | 4 | **8** | 12 | 84 GB |

Under ZeRO-1/2/3 the `KΨ` term is divided by `d`; the `2Ψ + 2Ψ` is divided by
`d` only at stages 2 and 3 respectively
(`parallelism-strategies/references/data-parallel-and-zero.md`).

## Why Adam costs `12` and what each byte does

```
fp32 master weight   4 bytes   the accumulator that survives small updates
first moment  m      4 bytes   EMA of the gradient
second moment v      4 bytes   EMA of the squared gradient
```

The update is `w ← w − lr · m̂/(√v̂ + ε)`. All three are per-parameter and none
can be reconstructed, so all three are resident.

## The fp32 master copy is a correctness requirement

Argued in full in
`gpu-architecture/references/tensor-cores-and-precision.md`. The short form:

bf16 carries 8 significand bits, so the smallest representable relative change
is `2⁻⁸ ≈ 0.0039`. A typical Adam update is `1e-3` to `1e-6` relative to the
weight. Updates below the rounding threshold are **discarded silently** —
`w + δ` rounds back to `w`. The run does not error; it stops learning.

fp32's 24 significand bits push the threshold to `2⁻²⁴ ≈ 6e-8`, six orders of
magnitude lower, which covers the entire realistic update range.

**Therefore: dropping the fp32 master weights to save `4Ψ` bytes is a
correctness change, not a memory optimization.** Present it as such. The
partial escapes:

- **Stochastic rounding** on the fp32→bf16 cast makes the *expected* update
  correct even when individual updates round away. Works well; needs kernel
  support.
- **Kahan summation** on a bf16 master keeps a bf16 compensation term — 2+2
  bytes, no saving over a 4-byte fp32 master, but it shards and offloads
  differently.

## Moments in bf16: usually fine

Unlike the master weight, the moments are *averages*, not accumulators of
small increments. Each step replaces a fraction of the buffer rather than
adding a tiny quantity to a large one, so bf16's precision is generally
adequate. `K` drops from 12 to 8 — a 25% cut in total model states — at low
risk.

The second moment `v` is the one to watch: it enters as `1/√v̂`, so relative
error there propagates directly into the effective per-parameter learning
rate. If a run destabilizes after switching to bf16 moments, `v` is the
suspect.

## 8-bit optimizers

Quantize `m` and `v` to 8 bits with **block-wise** quantization: split the
tensor into blocks (e.g. 2048 elements), store a per-block scale, and quantize
within the block using a non-linear (dynamic-exponent) codebook rather than
uniform levels.

Blockwise is the key detail. A single global scale would be destroyed by
outliers; per-block scales bound the dynamic range each quantizer must cover
and keep the error local. The reported result across many tasks is near-parity
with 32-bit optimizers.

```
K: 12 → 6      (4-byte fp32 master + 1 + 1)
7B model states: 112 GB → 70 GB
```

Keep the fp32 master. Quantizing the master is the correctness problem above;
quantizing the moments is a much milder approximation.

## Factored optimizers

Adafactor replaces the `h_in × h_out` second-moment matrix with its row and
column marginals — `O(h_in + h_out)` instead of `O(h_in · h_out)`. For a
transformer weight matrix that is a reduction from `h²` to `2h`, essentially
eliminating `v`.

The trade is real: the factored estimate is an approximation of the true
second moment, and results are model- and schedule-dependent. It also
interacts with the learning-rate schedule differently from Adam. It is common
in some large-scale training stacks and uncommon in others; treat it as a
choice to validate, not a free saving.

## Lion

Lion (arXiv:2302.06675, "Symbolic Discovery of Optimization Algorithms") keeps
a **single** momentum buffer and applies the sign of an interpolated momentum
as the update. `K` drops from 12 to 8 (one 4-byte buffer plus the fp32
master), a 33% cut in optimizer memory relative to Adam.

Practical notes: because the update is sign-based, its magnitude is uniform
across parameters, so the learning rate must be substantially smaller than
Adam's and weight decay correspondingly larger. It is a real memory saving
with a genuine hyperparameter retuning cost.

## Gradient memory

Gradients are `2Ψ` in bf16. Two ways they cost more than expected:

- **fp32 gradient accumulation.** Some stacks accumulate gradients in fp32 for
  numerical stability under large `m`, doubling this term to `4Ψ`. Whether it
  is needed depends on `m` and the loss scale; it is worth checking whether
  your stack does it silently.
- **A separate accumulation buffer.** If gradients are accumulated into a
  buffer distinct from `.grad`, both exist simultaneously.

Under ZeRO-2 and ZeRO-3 the gradient term is sharded to `2Ψ/d`, but note that
the *transient* full-precision gradient for the layer currently in backward
still exists — the peak is not exactly the steady-state figure.

## Choosing, when memory is the constraint

```
1. ZeRO-1 (shard KΨ by d).        Free — same 2Ψ traffic as DDP.
2. bf16 moments (K: 12 → 8).      ~free, low risk. Watch v.
3. 8-bit Adam (K: 12 → 6).        Small quality risk, well studied.
4. ZeRO-2, then ZeRO-3.           Free, then +50% comm.
5. CPU offload of the optimizer.  Capability, not performance.
6. A different optimizer (Lion, Adafactor).  Requires retuning — treat as an
                                  experiment, not a config change.
NEVER: drop the fp32 master weights to save memory (unless you have stochastic
       rounding). That is a correctness change.
```

## Sources

- ZeRO — arXiv:1910.02054
- ZeRO-Offload — arXiv:2101.06840
- Symbolic Discovery of Optimization Algorithms (Lion) — arXiv:2302.06675
