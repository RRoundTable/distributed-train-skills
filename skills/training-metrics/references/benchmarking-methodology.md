# Benchmarking Methodology

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`.

Most distributed-training performance disputes are measurement disputes. This
file is the minimum protocol that makes a number defensible.

> Platform recipe: launching benchmark jobs on a cluster and retrieving their
> output is skill `mlops:forge-train`. This file is about what to measure and
> how to compare.

## The minimum protocol

```
1. Warm up.        Discard at least the first 10 steps. More with torch.compile.
2. Measure ≥ 20 steady-state steps.
3. Report the MEDIAN step time, plus p90 and the min/max.
4. State: model, N (non-embedding), s, b, m, d, t, p, c, B, dtype,
          recomputation setting, n, GPU type, and the peak used for MFU.
5. Repeat the whole run at least twice. If the medians differ by more than
   a few percent, you have a placement or straggler problem, not a result.
```

Any number missing item 4 cannot be reproduced or compared, and should be
treated as an anecdote.

## Why warm-up matters more than people expect

The first steps include, in rough order of cost:

| Cost | Typical magnitude |
|---|---|
| `torch.compile` / Inductor compilation | seconds to minutes |
| cuBLAS / cuDNN algorithm autotuning | hundreds of ms per new shape |
| CUDA context creation | ~100s of ms, once |
| Caching-allocator growth (first `cudaMalloc`s) | tens of ms, decreasing |
| NCCL communicator setup on first collective | tens to hundreds of ms |
| Dataloader worker spin-up and first prefetch | varies |

Including them makes a 3-second step look like an 8-second step. Including
them *inconsistently* between two configs makes a comparison meaningless.

The reverse mistake also exists: measuring *only* steady state hides a real
cost when the job is short or restarts often. Report warm-up separately if it
matters; never fold it into the steady-state number.

## Timing correctly

CUDA is asynchronous. `time.time()` around a launch measures the launch, not
the work — the cost appears later at the next synchronizing call, attributed
to the wrong line.

Either synchronize explicitly before and after the region, or use CUDA events
recorded on the stream and read after a synchronize. In a distributed run,
also be aware that a synchronize before a collective is effectively a barrier
and will attribute straggler wait time to whichever rank you measured.

The practical consequence: **measure end-to-end step time**, which needs only
one synchronization per step, rather than fine-grained per-op timings that
each require a sync and perturb what they measure. Use a profiler for
attribution, not hand timers.

## Variance is data, not noise

| Pattern | Reading |
|---|---|
| p90 ≈ p50 | clean; the median is meaningful |
| p90 slightly above p50, regular | pipeline bubble or a periodic collective |
| p90 ≫ p50, occasional | a straggler, or checkpointing/eval landing in the window |
| Slow upward drift across steps | fragmentation, a leak, or thermal throttling |
| Bimodal | two different code paths (e.g. an eval step, or MoE routing skew) |
| Differs between otherwise identical runs | scheduler placement — the run is not the variable |

Reporting only a mean hides all six. Report median and p90 at minimum.

## Comparing two configurations

Hold everything except the variable fixed:

| Must be identical | Why |
|---|---|
| `B = b·m·d` | otherwise it is a different experiment |
| `s` | changes FLOPs per sample and the attention fraction |
| dtype and recomputation setting | changes step time and HFU/MFU |
| model and `N` | obviously, but check the vocab padding did not change |
| step range measured | same warm-up exclusion |
| data and seed, if loss is compared | otherwise the curves are not comparable |
| node placement, as far as possible | a different leaf switch is a different machine |

If the variable *is* `B` (e.g. testing gradient accumulation), then throughput
alone cannot decide — report tokens/s **and** loss vs tokens, because past the
critical batch size more throughput is not more progress
(`references/compute-budget-and-scaling-laws.md`).

## What to report

A defensible result is a table, not a sentence:

```
config          tokens/s   tokens/s/GPU   step p50   step p90   MFU    peak used
A (ZeRO-2)       124,300      15,540        8.44 s     8.61 s   41.2%  989 TF dense
B (ZeRO-3)       118,900      14,860        8.82 s     9.40 s   39.4%  989 TF dense
```

with the fixed parameters stated once above it. The `p90` column is what tells
a reader that config B has a variance problem as well as a mean problem.

## Micro-benchmarks and when they lie

Isolated benchmarks are useful for bounding a component and misleading for
predicting an end-to-end result.

| Micro-benchmark | Valid conclusion | Invalid conclusion |
|---|---|---|
| `nccl-tests` all-reduce busbw | the fabric's achievable bandwidth | "so my training will be this fast" — real collectives contend with compute |
| A single GEMM timed in isolation | that kernel's efficiency | end-to-end MFU — the gaps between kernels are missing |
| Attention kernel timing | the kernel's improvement | step-time improvement — attention may not be the bottleneck |
| Memory bandwidth benchmark | peak achievable | your kernel's achievable, which depends on access pattern |

The rule: a micro-benchmark establishes an **upper bound** on what a component
can contribute. It never establishes what it *will* contribute. Always close
the loop with an end-to-end measurement — a component-level win that does not
move step time was not on the critical path.

## Checklist before quoting a number

1. Warm-up excluded, ≥20 steps, median reported with p90?
2. All of item 4 in the protocol stated?
3. `P_peak` dense, at the training dtype, and named?
4. MFU or HFU said explicitly if recompute is on?
5. Run repeated, medians agree?
6. Comparison holds `B`, `s`, dtype, and data fixed?
7. Did the end-to-end number move, not just a component?
