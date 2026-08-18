---
name: gpu-architecture
description: |
  What happens inside a single GPU and what its hardware can do. Activate for:
  roofline, arithmetic intensity and ridge point, memory-bound vs compute-bound,
  peak FLOPs of A100 / H100 / B200 and achievable MFU on them, SM / occupancy,
  HBM vs L2 vs shared memory, tensor cores and tile quantization, bf16 / fp16 /
  fp8 / tf32 numerics, fp32 master weights, loss scaling, kernel fusion, why
  FlashAttention is faster, online softmax, torch.compile / Triton, Nsight
  profiling, "H100에서 7B 모델 MFU 얼마 나와야 정상이야".
  Do NOT activate for how the model is split across GPUs
  (parallelism-strategies), collectives or interconnect
  (communication-backends), HBM capacity budgeting/OOM/offload
  (memory-offloading), or end-to-end MFU measurement and loss curves
  (training-metrics). Do NOT activate for real cluster operations — submit,
  quota, which GPUs are free, job logs, nvidia-smi from a running job:
  mlops:forge-train.
---

# GPU Architecture

The machine one rank runs on: its peak rates, its memory hierarchy, its
numerics, and how to tell whether a kernel is limited by math or by memory.

> Platform recipe: how many GPUs are available, quota, which node type a job
> landed on, or reading `nvidia-smi` from a live job is skill `mlops:forge-train`.
> This skill explains what the hardware *can* do; it does not inspect a running
> system.

Symbols (`s`, `b`, `h`, `a`, `L`, `V`, `P_peak`) and bytes-per-element are
defined in skill `distributed-train:distributed-training-router`, file
`distributed-training-router/references/notation-and-glossary.md`.

## The roofline in one paragraph

A kernel's achievable performance is bounded by two things: the chip's peak
FLOP rate, and the rate at which it can feed operands from HBM. Plot
attainable FLOP/s against **arithmetic intensity** `AI = FLOPs / bytes moved`
and you get two straight lines:

```
attainable = min( P_peak ,  AI × BW_HBM )
```

Their intersection is the **ridge point** `AI* = P_peak / BW_HBM`. A kernel
with `AI < AI*` is **memory-bound** — it cannot reach peak no matter how good
the math is. A kernel with `AI > AI*` is **compute-bound**.

## Ridge points (dense, computed from published specs)

| GPU | Precision | Dense peak | HBM BW | Ridge `AI*` (FLOP/byte) |
|---|---|---|---|---|
| A100 40GB SXM | bf16/fp16 | 312 TFLOP/s | 1.555 TB/s | **201** |
| A100 80GB SXM | bf16/fp16 | 312 TFLOP/s | 2.039 TB/s | **153** |
| H100 SXM | bf16/fp16 | 989 TFLOP/s | 3.35 TB/s | **295** |
| H100 SXM | fp8 | 1979 TFLOP/s | 3.35 TB/s | **591** |
| B200 (DGX B200) | fp8 | 4500 TFLOP/s | 8 TB/s | **562** |

NVIDIA quotes H100 SXM at 1,979 TFLOPS bf16 and 3,958 TFLOPS fp8 **with
sparsity**; the dense numbers above are half of those, and dense is what dense
transformer training gets. DGX B200 is specified at 1,440 GB and 64 TB/s
across 8 GPUs (180 GB and 8 TB/s each) with FP8 at 72 PFLOPS sparse — 4.5
PFLOPS dense per GPU.

Two things follow immediately, and they are the most load-bearing facts in
this skill:

1. **The ridge point has been climbing.** A100-80 sits at 153; H100 at 295.
   Compute grew faster than bandwidth, so *more* kernels are memory-bound on
   newer hardware than on older. Optimizations that were unnecessary on A100
   become mandatory on H100.
2. **FP8 doubles the ridge point** (295 → 591 on H100) because it doubles
   FLOP/s while leaving bandwidth unchanged. Halving the bytes per element
   also halves the bytes a kernel moves, so the net effect depends on the
   kernel — but the *bar* for being compute-bound doubles.

## Where transformer operations sit

Approximate arithmetic intensity, bf16, ignoring caching:

| Operation | AI (FLOP/byte) | vs H100 ridge (295) | Bound by |
|---|---|---|---|
| Large GEMM, `M=N=K=8192` | ~2700 | ≫ | compute |
| QKV / output projection, `b·s = 8192` | ~1000–3000 | ≫ | compute |
| MLP up/down projection | ~1000–3000 | ≫ | compute |
| **Attention `QKᵀ` and `PV`** | **≈ head dim, 64–128** | **≪** | **memory** |
| Softmax | ~1 | ≪ | memory |
| LayerNorm / RMSNorm | ~1–2 | ≪ | memory |
| GeLU / SiLU (unfused) | ~1 | ≪ | memory |
| Residual add | ~0.5 | ≪ | memory |
| Optimizer step (Adam) | ~1–2 | ≪ | memory |
| Embedding lookup | ~0 | ≪ | memory |

The middle row is the whole story of the last several years. **Attention's
arithmetic intensity is approximately the head dimension** — 64 or 128 for
essentially every model — which is 2–5× *below* the H100 ridge point. Attention
is memory-bound by construction, and no amount of tensor-core throughput fixes
it. That is why FlashAttention exists, and why fusing it mattered more than
any single kernel optimization of the era.

Concretely, naive attention materializes the `b·a·s·s` score matrix in HBM. At
`b=1`, `a=32`, `s=8192`, bf16:

```
scores = 1 · 32 · 8192 · 8192 · 2 bytes = 4.3 GB    per layer, written then read
```

Written once, read for softmax, written again, read for `PV` — several times
4.3 GB of HBM traffic per layer, per forward, for an operation whose useful
FLOPs would take a fraction of the time. FlashAttention never materializes it.
See `references/kernel-fusion-and-flash-attention.md`.

## "GPU utilization" is the most misleading number in the stack

`nvidia-smi`'s utilization is the **fraction of sampling intervals in which at
least one kernel was resident**. It is not a fraction of peak FLOPs, not
occupancy, and not efficiency. A single memory-bound kernel that saturates
nothing reports 100%.

| Metric | What it measures | Useful for |
|---|---|---|
| `nvidia-smi` util | any kernel resident | is the GPU doing *anything* — liveness only |
| Occupancy | resident warps / max warps | latency hiding; high occupancy ≠ high throughput |
| Achieved FLOP/s | actual math rate | the real efficiency number |
| DRAM throughput | HBM bytes/s vs peak | whether you are on the memory roof |
| MFU | model FLOPs / (peak × time) | end-to-end truth — `distributed-train:training-metrics` |

If someone reports "GPU util is 100% so we're compute-bound", the correct
response is to ask for achieved FLOP/s or DRAM throughput. Those two together
place the kernel on the roofline; utilization does not.

## Precision

| dtype | bits | mantissa | exponent range | Role |
|---|---|---|---|---|
| fp32 | 32 | 23 | ±38 | master weights, optimizer moments, reductions |
| tf32 | 32 stored | 10 | ±38 | Ampere+ matmul input format; fp32 storage, faster math |
| bf16 | 16 | 7 | ±38 | default compute dtype — **same range as fp32** |
| fp16 | 16 | 10 | ±5 | more precision, far less range — needs loss scaling |
| fp8 e4m3 | 8 | 3 | ±2 | forward activations/weights |
| fp8 e5m2 | 8 | 2 | ±5 | gradients (needs the range) |

bf16 won for training because **range matters more than mantissa** for
gradients. fp16's exponent tops out around 65504 and underflows below ~6e-8,
so small gradients silently become zero — hence loss scaling. bf16 has fp32's
range and simply loses precision, which optimizers tolerate.

**Why fp32 master weights are not optional.** Adam's update is typically
`1e-3` to `1e-6` relative to the weight. bf16 has 8 significant bits, so
`w + δ` with `δ/w < 2⁻⁸ ≈ 0.004` rounds back to `w` — the update vanishes.
Keeping the master copy in fp32 (24 bits of significand, `2⁻²⁴ ≈ 6e-8`)
preserves updates six orders of magnitude smaller. This is what the fp32 term
in the `2Ψ + 2Ψ + KΨ` budget buys, and it is why "just store weights in bf16"
is a correctness change, not a memory optimization. Details in
`references/tensor-cores-and-precision.md`.

## Tile granularity — the free few percent

Tensor-core GEMMs execute in fixed tiles (typically 128×128 or 128×256 output
tiles built from 16×8×16 MMA fragments). A dimension that is not a multiple of
the tile forces a partially-empty final tile whose cost is the same as a full
one.

The canonical example: GPT-2's vocabulary is **50257**. Padding it to
**50304** — the next multiple of 64 — costs 47 unused embedding rows and
measurably improves throughput on the output projection and its gradient,
because 50257 is a prime-adjacent size that wastes most of its last tile. The
same argument applies to hidden sizes, FFN intermediate sizes, and per-rank
shard sizes after tensor parallelism: `h/t` should stay tile-aligned.

Check divisibility by 64 or 128 on every dimension that reaches a GEMM. It is
the cheapest optimization in this file.

## Diagnosing a slow kernel

```
1. Where is it on the roofline?
   achieved FLOP/s and DRAM throughput, both as % of peak
      high FLOP/s, low DRAM   → compute-bound. Optimize math or use lower precision.
      low FLOP/s, high DRAM   → memory-bound. Fuse, tile, or raise AI.
      low, low                → latency/occupancy-bound. Too few blocks, or stalls.
2. Is it one kernel or thousands of small ones?
      many tiny kernels → fusion (torch.compile) and launch overhead
3. Is the dominant cost attention?
      then AI ≈ head dim, and the answer is a fused attention kernel
4. Are dimensions tile-aligned?
5. Is it the first step? Exclude warm-up before concluding anything.
```

## Rules for answering here

- Place a claim on the roofline before recommending an optimization. "Make it
  faster" without naming the roof is a guess.
- Use **dense** peaks unless sparsity is genuinely in use, and say which.
  Vendor headline numbers are usually the sparse ones.
- Distinguish utilization from achieved FLOP/s every time either appears.
- State the precision when quoting any peak.
- Do not emit `forge` commands or inspect a live cluster.

## Reference files

| File | Contents |
|---|---|
| `references/memory-hierarchy.md` | registers/SMEM/L1/L2/HBM, capacities and bandwidths, occupancy, coalescing, microbenchmark-derived cache numbers |
| `references/roofline-and-arithmetic-intensity.md` | the model derived, ridge-point table, AI for every transformer op, worked examples, the hierarchical roofline |
| `references/tensor-cores-and-precision.md` | MMA shapes, tile quantization, dtype anatomy, loss scaling, fp8 scaling, the fp32 master-weight argument |
| `references/kernel-fusion-and-flash-attention.md` | fusion economics, online softmax derived, FlashAttention 1/2/3, torch.compile and Triton |
| `references/profiling-with-nsight.md` | Nsight Systems vs Compute vs the PyTorch profiler, what to measure, reading a timeline, common misreadings |
