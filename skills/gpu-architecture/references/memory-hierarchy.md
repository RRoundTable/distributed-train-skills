# Memory Hierarchy

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`.

## The levels

Per GPU. Capacities are per-SM where noted; bandwidths are aggregate.

| Level | Capacity | Bandwidth | Latency | Scope |
|---|---|---|---|---|
| Registers | 256 KB per SM (64K × 32-bit) | ~100 TB/s aggregate | ~1 cycle | one thread |
| Shared memory / L1 | up to 228 KB per SM (H100) | ~10–20 TB/s aggregate | ~30 cycles | one thread block |
| L2 cache | 50 MB (H100), 40 MB (A100) | several TB/s | ~200 cycles | whole GPU |
| HBM | 80 GB (H100 SXM), 180 GB (B200) | 3.35 TB/s (H100), 8 TB/s (B200) | ~400–600 cycles | whole GPU |
| Host DRAM over PCIe Gen5 | TB scale | ~64 GB/s bidir. | ~µs | one node |

Each step down is roughly an order of magnitude in bandwidth and latency. The
entire craft of GPU kernel optimization is keeping data at the highest level
that fits, for as long as possible.

## Microbenchmark-derived cache numbers

Vendors publish HBM bandwidth but not L1/L2 bandwidth, so those come from
microbenchmarks. The figures below are measured on an **H800** (not an H100 —
the H800 is the bandwidth-reduced variant, though the on-die cache hierarchy
is the same architecture) and an A100, reported in bytes per clock:

| Level | A100 | H800 |
|---|---|---|
| L1 / shared memory | 106.8 B/clk/SM | 125.8 B/clk/SM |
| L2 | 2007.9 B/clk | 4472.3 B/clk |
| Global (measured) | 1407.2 GB/s | 1861.5 GB/s |

Source: "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture",
arXiv:2402.13499. Converting bytes/clock to GB/s requires a clock assumption;
at ~1.7 GHz the H800 L2 figure corresponds to roughly 7.6 TB/s aggregate —
about 4× its measured HBM bandwidth. Treat that conversion as approximate and
the bytes/clock figures as the primary data.

The same work measures dense `wgmma` tensor-core throughput on H800 at
approximately **1448 TFLOPS for FP8** and **729 TFLOPS for BF16**, exceeding
95% of theoretical peak — a useful reminder that the peak numbers are
reachable by a well-written kernel, so a large gap in your own kernel is
yours, not the hardware's.

## Why L2 matters more than people expect

L2 is ~4× HBM bandwidth and 50 MB on H100 — large enough to hold meaningful
working sets. Two consequences:

- **A kernel that fits its working set in L2 has a much higher effective
  bandwidth than the roofline's HBM line suggests.** This is why measured
  performance sometimes exceeds a naive roofline prediction, and why the
  hierarchical roofline (a separate roof per memory level) is the more honest
  model — see `references/roofline-and-arithmetic-intensity.md`.
- **Back-to-back kernels can be much cheaper than their sum** if the second
  reads what the first wrote and it is still in L2. This is a large part of
  why kernel fusion pays off even when each kernel is individually
  "memory-bound at HBM rates".

## Shared memory and the tiling pattern

Shared memory is programmer-managed cache. The universal GEMM structure:

```
for each output tile:
    load a tile of A and a tile of B from HBM into shared memory   (coalesced)
    __syncthreads()
    each thread computes a sub-tile from shared memory into registers
    __syncthreads()
    write the output tile back to HBM
```

The arithmetic intensity of this structure is set by the tile size. For an
`M×N×K` GEMM with `T×T` output tiles, each element of A and B is read from
HBM roughly `N/T` and `M/T` times respectively, so **larger tiles → higher
AI**. Tile size is bounded by shared memory capacity and register pressure —
which is precisely why 228 KB of SMEM per SM on H100 (up from 164 KB on A100)
was an architecturally significant change, and why FlashAttention's block
sizes are chosen against that capacity.

## Occupancy: necessary, not sufficient

Occupancy = resident warps per SM / maximum resident warps. High occupancy
gives the scheduler more warps to switch to while one stalls on memory, which
hides latency.

But **high occupancy does not mean high throughput**:

- A memory-bound kernel at 100% occupancy is still bound by HBM bandwidth.
- Very high-performing GEMM kernels often run at *low* occupancy on purpose,
  because each thread holds a large register-resident accumulator tile — more
  registers per thread means fewer resident threads, and the register-blocked
  arithmetic intensity is worth far more than the extra latency hiding.

Occupancy is limited by whichever of these binds first: registers per thread,
shared memory per block, block size, or the hardware warp limit. When
Nsight Compute reports an occupancy limiter, it is telling you which resource
to trade, not that you should maximize occupancy.

## Coalescing and access patterns

HBM is accessed in transactions (128-byte cache lines / 32-byte sectors).
When the 32 threads of a warp read 32 consecutive 4-byte values, the hardware
services them in a small number of transactions. When they read with a large
stride, each thread's read may pull its own line, and effective bandwidth
falls by up to 32×.

| Pattern | Efficiency |
|---|---|
| Consecutive (`thread i` reads `base + i`) | full |
| Strided by `k` elements | ~`1/min(k, 32)` |
| Random / gather | worst; also defeats prefetch |
| Broadcast (all threads read the same address) | full — served once |

For transformer workloads the practical implications are about **layout**:
a `[batch, seq, heads, dim]` tensor and a `[batch, heads, seq, dim]` tensor
have identical contents and very different kernel performance, which is why
attention implementations are opinionated about layout and why an unnecessary
`transpose` + `contiguous` in a hot path is expensive.

**Bank conflicts** are the shared-memory analogue: SMEM is divided into 32
banks, and two threads in a warp hitting different addresses in the same bank
serialize. The standard fix is padding a tile's row stride by one element so
that column-wise access walks across banks rather than down one.

## Vectorized access

Loading 128 bits per thread (`float4`, or 8 bf16 values) rather than 32 bits
reduces the number of memory instructions by 4× and is often the difference
between reaching 60% and 90% of HBM bandwidth on a bandwidth-bound kernel.
The microbenchmark numbers above are for vectorized (`.v4`) access for exactly
this reason. This requires alignment — another argument for keeping tensor
dimensions multiples of 8 or 16 elements.

## Capacity is a different skill

This file is about **bandwidth and access**. How many bytes your model,
gradients, optimizer state, and activations occupy in HBM — and what to do
when they do not fit — is `distributed-train:memory-offloading`, starting from
`memory-offloading/references/memory-taxonomy-and-math.md`. The two interact at exactly one
point: the allocator. Fragmentation is a capacity problem caused by an
allocation-pattern problem, and it is covered there.

## Sources

- Benchmarking and Dissecting the Nvidia Hopper GPU Architecture — arXiv:2402.13499
- NVIDIA Hopper architecture overview —
  https://resources.nvidia.com/en-us-hpc-ai/nvidia-hopper-architecture
