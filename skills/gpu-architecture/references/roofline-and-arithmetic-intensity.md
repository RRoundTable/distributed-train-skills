# Roofline and Arithmetic Intensity

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`.

## The model

For a kernel performing `F` FLOPs while moving `Q` bytes to and from HBM:

```
arithmetic intensity   AI = F / Q          [FLOP per byte]
time                    T ≥ max(F/P_peak, Q/BW)
attainable rate         P = min(P_peak, AI · BW)
```

The two branches of that `min` are the two roofs. Their intersection is the
**ridge point**:

```
AI* = P_peak / BW
```

- `AI < AI*` → memory-bound. The kernel is starved. Peak FLOP/s is
  unreachable regardless of how good the math is.
- `AI > AI*` → compute-bound. Bandwidth is not the limit; the math or the
  instruction mix is.

## Ridge points, computed

Dense peaks (not sparsity-inflated) divided by published HBM bandwidth:

| GPU | Precision | Dense peak (TFLOP/s) | BW (TB/s) | `AI*` |
|---|---|---|---|---|
| V100 SXM2 | fp16 | 125 | 0.900 | 139 |
| A100 40GB SXM | bf16 | 312 | 1.555 | **201** |
| A100 80GB SXM | bf16 | 312 | 2.039 | **153** |
| H100 SXM | bf16 | 989 | 3.35 | **295** |
| H100 SXM | fp8 | 1979 | 3.35 | **591** |
| B200 (DGX B200) | fp8 | 4500 | 8.0 | **562** |

Three readings:

1. **The A100 80GB has a *lower* ridge point than the 40GB.** Same compute,
   more bandwidth. More kernels are compute-bound on the 80GB card. Memory
   capacity upgrades often bring bandwidth upgrades, and that changes the
   optimization calculus, not just what fits.
2. **The ridge point rose from 153 (A100-80) to 295 (H100).** Compute grew
   ~3.2× while bandwidth grew ~1.6×. **More kernels are memory-bound on
   newer hardware.** A kernel that was compute-bound on A100 can be
   memory-bound on H100 without changing a line of code — and its MFU will
   drop accordingly.
3. **FP8 doubles the ridge point** (295 → 591): peak doubles, bandwidth does
   not. An op only benefits fully from fp8 if it stays above the *new*, higher
   ridge. Halving element size also halves `Q` for that op, which raises its
   AI — so fp8 helps memory-bound ops through the byte count and compute-bound
   ops through the FLOP rate, but by different mechanisms.

## Arithmetic intensity of a GEMM

For `C = A·B` with `A: M×K`, `B: K×N`, `C: M×N`, in `p` bytes per element:

```
F = 2·M·N·K
Q = p·(M·K + K·N + M·N)               (assuming each is read/written once)
AI = 2MNK / (p·(MK + KN + MN))
```

For a square `M = N = K = n` in bf16 (`p = 2`):

```
AI = 2n³ / (2·3n²) = n/3
```

So a square GEMM's AI is one third of its dimension. On H100 bf16
(`AI* = 295`), a square GEMM is compute-bound once `n > 885`. Transformer
GEMMs have `n` in the thousands, so **all the big matmuls are comfortably
compute-bound** — and their MFU should be high.

The interesting cases are the ones with a small dimension:

| Shape | AI (bf16) | H100 verdict |
|---|---|---|
| `8192 × 8192 × 8192` | ~1365 | compute-bound |
| `8192 × 8192 × 128` | ~126 | **memory-bound** |
| `1 × 8192 × 8192` (batch-1 matvec) | ~1 | severely memory-bound |
| `8192 × 50304 × 8192` (output proj.) | ~1400 | compute-bound |

The second row is the general lesson: **a GEMM with any small dimension is
memory-bound.** Small micro-batch, small per-rank shard after aggressive
tensor parallelism, or a head dimension of 128 all land here.

## Why attention is memory-bound

For one attention head with sequence `s` and head dimension `d_h`:

```
QKᵀ:   F = 2·s²·d_h      Q ≈ p·(2·s·d_h + s²)     ← s² scores written out
PV:    F = 2·s²·d_h      Q ≈ p·(s² + s·d_h + s·d_h)
```

When `s ≫ d_h`, the `s²` terms dominate `Q` and:

```
AI ≈ 2·s²·d_h / (p·s²) = 2·d_h/p = d_h    (for bf16, p = 2)
```

```
AI_attention ≈ head dimension ≈ 64 to 128
```

Against an H100 ridge point of 295, that is **2.3× to 4.6× below the ridge**.
Attention is memory-bound by construction, for every model anyone trains, and
it gets *worse* on newer hardware as the ridge climbs.

This is the entire motivation for FlashAttention. It does not reduce FLOPs —
it reduces `Q` by never writing the `s²` score matrix to HBM, tiling the
computation so the scores live in shared memory. AI rises by roughly the tile
factor, moving attention across the ridge. Derivation in
`references/kernel-fusion-and-flash-attention.md`.

### The number that makes it concrete

Naive attention materializes a `b × a × s × s` score tensor. At `b=1`,
`a=32`, `s=8192`, bf16:

```
1 · 32 · 8192 · 8192 · 2 bytes = 4.295e9 bytes = 4.3 GB     per layer
```

That tensor is written by `QKᵀ`, read and written by softmax, and read by
`PV` — well over 10 GB of HBM traffic per layer per forward pass, for one
sample. At 3.35 TB/s that is ~3 ms per layer of pure memory traffic; across 32
layers, ~100 ms per forward, before any useful math. And it is the reason
naive attention OOMs at long context (`memory-offloading`).

## Every transformer op, placed

Approximate AI in bf16, for typical training shapes:

| Op | FLOPs | Bytes | AI | H100 (295) |
|---|---|---|---|---|
| QKV projection | `6·b·s·h²` | `~2·(b·s·h·4 + 3h²)` | ~1000+ | compute |
| `QKᵀ` | `2·b·a·s²·d_h` | `~2·b·a·s²` | `≈ d_h` | **memory** |
| softmax | `~5·b·a·s²` | `~4·b·a·s²` | ~1 | **memory** |
| `PV` | `2·b·a·s²·d_h` | `~2·b·a·s²` | `≈ d_h` | **memory** |
| output projection | `2·b·s·h²` | similar to QKV | ~1000+ | compute |
| MLP up (`h → 4h`) | `8·b·s·h²` | `~2·(b·s·5h + 4h²)` | ~1000+ | compute |
| activation (GeLU/SiLU) | `~10·b·s·4h` | `~2·2·b·s·4h` | ~2.5 | **memory** |
| MLP down (`4h → h`) | `8·b·s·h²` | similar | ~1000+ | compute |
| LayerNorm / RMSNorm | `~10·b·s·h` | `~2·2·b·s·h` | ~2.5 | **memory** |
| residual add | `b·s·h` | `~2·3·b·s·h` | ~0.17 | **memory** |
| Adam step | `~10·Ψ` | `~4·(2+4+4+4)Ψ` | ~0.2 | **memory** |
| embedding lookup | ~0 | `2·b·s·h` | ~0 | **memory** |

The pattern: **the four big GEMMs are compute-bound; everything else is
memory-bound.** Since the GEMMs carry the overwhelming majority of FLOPs, MFU
can still be high — but every memory-bound op between them is time in which
the tensor cores are idle. That is what kernel fusion recovers, and it is why
`torch.compile` on the non-GEMM ops is worth double-digit percentages.

The Adam row deserves attention: the optimizer step reads and writes
~14 bytes per parameter for ~10 FLOPs. For a 7B model that is ~100 GB of HBM
traffic per step at AI ≈ 0.2 — pure bandwidth, and a reason fused/multi-tensor
optimizers exist.

## The hierarchical roofline

The single-roof model assumes every byte comes from HBM. In reality there is a
roof per level:

```
P = min(P_peak, AI_HBM · BW_HBM, AI_L2 · BW_L2, AI_SMEM · BW_SMEM)
```

Where `AI_L2 = F / (bytes from L2)`, and so on. A kernel whose working set
fits in L2 (50 MB on H100) can exceed the HBM roofline, which is:

- why fused kernels beat their unfused equivalents by more than the FLOP count
  predicts,
- why measured performance sometimes lands *above* a naive roofline prediction
  (the model was not wrong, the assumed `Q` was),
- and why FlashAttention's speedup exceeds what a pure HBM-traffic accounting
  suggests — it also gets SMEM reuse.

## Using the roofline in practice

1. Get **achieved FLOP/s** and **DRAM throughput** for the kernel (Nsight
   Compute, or the PyTorch profiler's kernel view).
2. Compute `AI = achieved FLOP/s ÷ DRAM bytes/s`.
3. Compare to `AI*` for the GPU and precision.

| Result | Meaning | Action |
|---|---|---|
| `AI > AI*`, FLOP/s near peak | healthy compute-bound kernel | nothing to do |
| `AI > AI*`, FLOP/s far below peak | not bandwidth-limited but still slow | occupancy, tail effect, tile misalignment, instruction mix |
| `AI < AI*`, DRAM near peak | healthy memory-bound kernel | reduce bytes: fuse, lower precision, better layout |
| `AI < AI*`, DRAM far below peak | latency-bound | uncoalesced access, too few blocks, dependent stalls |
| both far below peak | launch/occupancy-bound | kernel too small; fuse or batch |

The fourth row is the one people misdiagnose most: a kernel can be
memory-bound *and* not achieving memory bandwidth, in which case fusing it
will not help until the access pattern is fixed.

## Connecting to MFU

The roofline is per kernel; MFU is per step. The bridge:

```
MFU ≈ Σ_kernels (model FLOPs in kernel) / (P_peak · T_step · n)
```

A step made entirely of compute-bound GEMMs at 80% of peak would show ~80%
MFU. Real steps mix in memory-bound ops, communication, and bubbles, which is
why the best published runs land at 38–55% (see
`training-metrics/references/flop-counting-and-mfu.md`). Working backwards
from MFU to a roofline diagnosis is exactly the router's separating test.
