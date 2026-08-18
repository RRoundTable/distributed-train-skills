# Profiling

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`.

> Platform recipe: launching a profiled job on a cluster, retrieving the trace
> file, and any environment needed to enable profiling there is skill
> `mlops:forge-train`. This file is about **what to measure and how to read
> it**.

## Pick the right tool for the question

| Question | Tool |
|---|---|
| Where does the step time go? Is comm overlapped? Are there gaps? | **Nsight Systems** (`nsys`) — timeline, whole process |
| Why is *this* kernel slow? | **Nsight Compute** (`ncu`) — per-kernel counters, roofline |
| Which PyTorch op maps to which kernel? What memory did it allocate? | **PyTorch profiler** — op-level attribution, exports Chrome traces |
| How much HBM is allocated over time, and where did it fragment? | **PyTorch memory snapshot** (`torch.cuda.memory._record_memory_history`) — see `memory-offloading` |
| Is the GPU alive at all? | `nvidia-smi` — liveness only, nothing more |

The order to use them: Nsight Systems first (where is the time?), then either
Nsight Compute (a kernel is slow) or the memory snapshot (memory is the
problem). Starting with per-kernel counters before knowing whether kernels are
even the bottleneck is the most common wasted profiling session.

## Rules that make a profile trustworthy

1. **Skip warm-up.** The first steps include CUDA context creation, cuBLAS
   autotuning, allocator growth, and `torch.compile`. Profile steps 10–20, not
   0–10.
2. **Profile a few steps, not one.** Step-to-step variance is signal — it
   distinguishes a straggler from a bandwidth limit.
3. **Synchronize before timing anything by hand.** CUDA is asynchronous;
   `time.time()` around a kernel launch measures the launch, not the work.
   Use CUDA events or the profiler.
4. **Profile the shape you actually run.** Halving the batch to make the trace
   smaller changes which roof you are on.
5. **On multi-rank runs, profile more than rank 0.** Rank 0 often does extra
   work (logging, checkpointing) and is not representative; and a straggler is
   by definition on another rank.

## Reading a Nsight Systems timeline

The rows that matter: CUDA kernels, memory copies, NCCL kernels, and the CPU
(Python/launch) row.

| Pattern | Diagnosis | Owner |
|---|---|---|
| Kernels back to back, few gaps | healthy | — |
| Wide gaps, CPU row busy | CPU-bound launch; too many small ops | fuse / `torch.compile` |
| Wide gaps, CPU row idle too | a host sync — `.item()`, `.cpu()`, `print(loss)` | remove it |
| NCCL kernels interleaved with compute | overlap working | — |
| One long NCCL block after all backward compute | overlap broken | `communication-backends/references/overlap-and-scheduling.md` |
| NCCL between every micro-batch | missing `no_sync()` under accumulation | same |
| Regular idle at fixed offsets, same every step | pipeline bubble | `parallelism-strategies/references/pipeline-parallel.md` |
| Irregular idle, different rank late each step | straggler | `communication-backends/references/hang-and-straggler-debugging.md` |
| `Memcpy DtoH` inside the loop | an unintended host transfer or offload | `memory-offloading` |
| Thousands of sub-10µs kernels | launch-bound | fuse |

The single highest-value read is the ratio of **kernel-busy time to wall time**
for one step. If kernels occupy 60% of the step, the other 40% is gaps, and no
kernel optimization will recover it.

## Reading Nsight Compute for one kernel

The four numbers to extract, in order:

1. **Duration** — is this kernel actually worth optimizing? Sort by total time
   (duration × invocations), not by duration.
2. **Achieved FLOP/s** as a percentage of peak.
3. **DRAM throughput** as a percentage of peak.
4. **Occupancy** and its limiter.

Then place it on the roofline (`references/roofline-and-arithmetic-intensity.md`):

| FLOP/s | DRAM | Verdict | Next step |
|---|---|---|---|
| high | low | compute-bound, healthy | lower precision, or accept |
| low | high | memory-bound, healthy | fuse, improve layout, lower precision |
| low | low | latency/occupancy-bound | uncoalesced access, too few blocks, dependent stalls |
| high | high | near both roofs | done |

Nsight Compute also draws the roofline directly, which is faster than
computing `AI` by hand — but knowing the ridge point for your GPU and
precision from the table in the roofline file lets you sanity-check what it
draws.

**Warning:** `ncu` serializes and replays kernels to collect counters. It is
enormously slower than the real run and it perturbs anything timing-dependent.
Never use it to measure end-to-end step time; use it only for per-kernel
counters.

## The PyTorch profiler

Its advantage over the Nsight tools is **attribution**: it maps GPU kernels
back to the Python operator and, with stack recording, the source line. That
is what you want when the question is "which part of my model is this
kernel?", not "how fast is this kernel".

Useful configuration in practice:

- Record shapes, so you can spot a GEMM with a badly-aligned dimension.
- Record memory, so allocation spikes line up with ops.
- Use the scheduler to skip warm-up and capture a few active steps, otherwise
  the trace is enormous and dominated by startup.
- Export the Chrome trace and read it in a trace viewer; the table view hides
  the gaps, which are usually the story.

## Interpreting `nvidia-smi`

`nvidia-smi` utilization is the fraction of sampled intervals in which **any**
kernel was resident. It says nothing about efficiency. Concretely:

- A memory-bound elementwise kernel running alone → 100% utilization, ~2% of
  peak FLOP/s.
- A well-optimized GEMM → 100% utilization, ~75% of peak FLOP/s.

These are indistinguishable in `nvidia-smi` and completely different
situations. Utilization answers exactly one question — *is the GPU doing
anything?* — which is worth knowing when a job appears hung, and worth nothing
otherwise.

The genuinely useful `nvidia-smi` fields are **memory used** (capacity
pressure), **power draw** (a chip at 30% of TDP is not doing heavy math),
**SM clock** (throttling), and **temperature** (a straggler cause).

## A profiling workflow that converges

```
1. Measure step time and MFU first.        → training-metrics
   Without these you cannot tell whether anything is wrong.
2. Nsight Systems on steps 10–20, several ranks.
   → What fraction of the step is kernels? What fraction is NCCL? Gaps?
3. Branch on the answer:
      gaps                → CPU-bound or host sync. Fix that first; it distorts everything.
      NCCL-dominated      → communication-backends
      regular idle blocks → parallelism-strategies (bubble)
      kernel-dominated    → continue
4. Sort kernels by total time. Take the top 3.
5. Nsight Compute on those. Place each on the roofline.
6. Fix the top one. Re-measure MFU end to end.
```

Step 6 is the one people skip. A kernel-level improvement that does not move
end-to-end MFU either was not on the critical path or was offset elsewhere;
either way it is not a win yet.

## Common misreadings

| Misreading | Reality |
|---|---|
| "GPU util is 100%, we're compute-bound" | utilization measures residency, not efficiency |
| "The profile says NCCL takes 40% of the time" | if it overlaps compute, that time is not additive — check concurrency, not just totals |
| "This kernel is 90% of the time" — from a single unsynchronized timer | asynchronous launch; the time landed on the next synchronizing call |
| "Occupancy is 40%, that's the problem" | high-performance GEMMs often run at low occupancy by design |
| "The first step took 8 seconds" | warm-up, compilation, autotuning |
| "Rank 0's profile shows the bottleneck" | rank 0 does extra work; the straggler is elsewhere |
| "Adding more GPUs will fix this kernel" | a per-GPU kernel inefficiency is unchanged by world size |
