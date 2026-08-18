# Tensor Cores and Precision

Symbols and bytes-per-element: router → `distributed-training-router/references/notation-and-glossary.md`.

## What a tensor core does

A tensor core executes a small matrix multiply-accumulate in one instruction:
`D = A·B + C`, where `A` and `B` are low precision and the accumulation `C`,
`D` is higher precision. The programmer-visible MMA shapes are small (e.g.
`m16n8k16` for bf16 on Ampere/Hopper); libraries compose them into the
128×128-ish output tiles that a warp group actually computes.

Hopper adds `wgmma` — warpgroup-level asynchronous MMA — which lets a group of
four warps issue a much larger MMA that overlaps with memory movement (via the
Tensor Memory Accelerator). Microbenchmarks put dense `wgmma` above 95% of
theoretical peak on H800: ~1448 TFLOPS fp8 and ~729 TFLOPS bf16
(arXiv:2402.13499). The hardware is reachable; a large gap in your own kernel
is a software gap.

The key asymmetry: **inputs are low precision, accumulation is fp32**. This is
what makes bf16 training numerically viable — the products are computed in
bf16 but summed in fp32, so a length-`K` dot product does not accumulate bf16
rounding error `K` times.

## Tile quantization

Because the hardware works in tiles, a GEMM dimension that is not a multiple
of the tile size wastes the remainder of the final tile — which costs the same
as a full one.

```
tiles needed = ceil(M/T_m) · ceil(N/T_n)
efficiency   = (M·N) / (ceil(M/T_m)·T_m · ceil(N/T_n)·T_n)
```

At `T = 128` and `N = 129`, you pay for 256 columns to compute 129 — 50%
waste on that dimension.

**The canonical example.** GPT-2's vocabulary is **50257**. The output
projection is `[b·s, h] × [h, V]`, so `V` is a GEMM dimension. Padding
`V = 50257 → 50304` (the next multiple of 64, and also of 128) adds 47 unused
rows — a rounding error in parameter count — and produces a measurable
throughput improvement on the output projection and its backward. The unused
logits are masked out of the loss, so nothing changes numerically.

Dimensions to check for tile alignment:

| Dimension | Should be a multiple of |
|---|---|
| vocabulary `V` | 64 or 128 (and of `t` for vocab parallelism) |
| hidden size `h` | 128 |
| FFN intermediate size | 128 |
| head dimension `d_h` | 64 (usually already 64/128) |
| **per-rank shard `h/t`** | 128 — TP can break alignment that the full model had |
| batch × sequence `b·s` | 128 where practical |

The per-rank row is the one that catches people: `h = 8192` with `t = 8` gives
`1024` (fine), but an FFN intermediate of `11008` with `t = 8` gives `1376`,
which is a multiple of 32 but not of 128.

There is also a **wave quantization** effect one level up: the GPU runs tiles
in waves across SMs, so a GEMM producing 133 tiles on a 132-SM chip runs two
waves and takes nearly twice as long as one producing 132. This shows up as a
step-function in throughput versus batch size and is a reason to benchmark a
few nearby shapes rather than assume monotonicity.

## The dtypes

| dtype | sign/exp/mant | Max | Min normal | Notes |
|---|---|---|---|---|
| fp32 | 1/8/23 | 3.4e38 | 1.2e-38 | reference |
| tf32 | 1/8/10 | 3.4e38 | 1.2e-38 | fp32 storage; Ampere+ matmul input |
| bf16 | 1/8/7 | 3.4e38 | 1.2e-38 | fp32's range, 8 significant bits |
| fp16 | 1/5/10 | 65504 | 6.1e-5 | more mantissa, **far** less range |
| fp8 e4m3 | 1/4/3 | 448 | 2ᵉ… | forward pass |
| fp8 e5m2 | 1/5/2 | 57344 | | gradients — needs the range |

**Why bf16 beat fp16 for training.** Gradients span many orders of magnitude
and routinely fall below fp16's smallest normal value (6.1e-5), where they
flush to zero — silently. bf16 has fp32's exponent range and simply carries
fewer significant bits, which optimizers tolerate because the update is
averaged over a large batch anyway. bf16 needs no loss scaling; fp16 does.

**Loss scaling** (fp16 only): multiply the loss by a large factor `S` before
backward so gradients land in fp16's representable range, then divide by `S`
before the optimizer step. Dynamic loss scaling raises `S` when no overflow
occurs for a while and halves it on an `inf`/`NaN`, skipping that step. A
run that skips many steps has a scaler oscillating around the overflow
threshold — visible as a sawtooth in the logged scale and as steps whose loss
does not change.

**tf32** is a stealth default: on Ampere and later, PyTorch may run fp32
matmuls with tf32 inputs (10 mantissa bits) unless told otherwise. It is
usually the right trade, but it means "fp32 training" is often not what it
says, and it is a common source of small numerical differences between an
A100 run and an older baseline.

## Why fp32 master weights are not optional

This is the argument behind the `KΨ` term in the model-state budget
(`parallelism-strategies/references/data-parallel-and-zero.md`).

bf16 has 8 bits of significand, so the smallest relative change it can
represent is `2⁻⁸ ≈ 0.0039`. An Adam update is typically
`lr × (m̂/(√v̂+ε))`, with a relative magnitude of `1e-3` to `1e-6` against the
weight. Then:

```
w_bf16 + δ  with  δ/w < 2⁻⁸   →   rounds back to w.  The update is lost.
```

At `lr = 1e-4` and a typical normalized update, most parameters' updates fall
below the bf16 rounding threshold. The run does not crash; it just stops
learning in a way that looks like a bad learning rate.

fp32 has 24 bits of significand, so the threshold is `2⁻²⁴ ≈ 6e-8` — six
orders of magnitude smaller. Keeping the master copy in fp32, updating it, and
casting down to bf16 for the next forward is what makes mixed precision work.

Alternatives that partially avoid the cost:

- **Stochastic rounding** when casting fp32→bf16 makes the *expected* update
  correct even when individual updates round away. Effective, but needs
  hardware or kernel support and is not universal.
- **Kahan summation** on a bf16 master weight keeps a bf16 compensation term:
  4 bytes per parameter instead of fp32's 4 — no saving over fp32 master, but
  it composes differently with sharding.
- **bf16 optimizer moments** (`K = 4` instead of 12) is common and mostly
  safe; the moments are averages, not accumulating quantities.

Dropping the fp32 master copy to save `4Ψ` bytes is a **correctness** change,
not a memory optimization, and should be presented as such.

## FP8 training

FP8 is not a drop-in dtype swap. Its dynamic range is tiny (e4m3 maxes at
448), so every tensor needs a **scaling factor** that keeps its values inside
the representable window:

```
x_fp8 = quantize(x / scale)      matmul in fp8      dequantize with the scales
```

The engineering is in choosing `scale`. Delayed scaling keeps a history of
recent absolute maxima per tensor and picks a scale from it, which avoids a
synchronization to compute the current max but reacts a step late — so a
sudden activation spike overflows. Per-tensor scaling is the standard
granularity; finer (per-block) scaling is more robust and more expensive.

Practical division of labour that has emerged: **e4m3 for forward** (weights
and activations, where precision matters more than range) and **e5m2 for
gradients** (where range matters more). Master weights and optimizer state
stay fp32 regardless.

Note the roofline consequence from
`references/roofline-and-arithmetic-intensity.md`: fp8 doubles peak FLOP/s
without changing bandwidth, so the ridge point doubles (295 → 591 on H100).
An op that was marginally compute-bound in bf16 can be memory-bound in fp8,
and the speedup will be far below 2×.

## Numerical debugging quick reference

| Symptom | Likely cause |
|---|---|
| `NaN` in fp16, fine in bf16 | fp16 range — loss scaling misconfigured, or switch to bf16 |
| Loss stops improving, no error | bf16 master weights swallowing updates, or lr too small |
| Loss scale oscillates / many skipped steps | dynamic scaler at the overflow threshold |
| Results differ slightly between GPUs | tf32 enabled, or nondeterministic reduction order |
| fp8 run diverges after a stable start | scaling factors stale (delayed scaling) at an activation spike |
| Gradient norm is `inf` on one rank only | that rank's data, not a precision issue |
| Small numeric diffs vs an older baseline | tf32 default, or a different cuBLAS/cuDNN algorithm choice |

## Sources

- Benchmarking and Dissecting the Nvidia Hopper GPU Architecture — arXiv:2402.13499
- NVIDIA Hopper architecture overview —
  https://resources.nvidia.com/en-us-hpc-ai/nvidia-hopper-architecture
- GLM-130B (a documented case study in large-scale precision/stability
  problems and their mitigations) — arXiv:2210.02414
