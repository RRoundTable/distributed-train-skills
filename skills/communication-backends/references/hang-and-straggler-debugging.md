# Hangs and Stragglers

> **Deferral banner.** If you are looking at a **real job on a real cluster** —
> its logs, its NCCL error output, its exit code, its scheduler state, or the
> environment variables it needs on that cluster — that is skill
> `mlops:forge-train` → `error-patterns.md` §3 (NCCL). This file explains the
> *mechanism* and the *localization procedure*, so that the log you retrieve
> there means something. It contains no cluster-specific values.

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`.

## The central insight

**Every collective is a barrier.** `all_reduce`, `all_gather`,
`reduce_scatter`, and `barrier` do not return on *any* rank until *every*
rank in the process group has called them with compatible arguments.

Therefore:

```
A collective timeout does not mean the network is slow.
It means one or more ranks never arrived at that collective.
```

This single reframing changes what you do next. The question is not "why is
communication slow" but "**which rank is missing, and what is it doing
instead?**" Raising the timeout converts a 10-minute failure into an hour-long
one and discards the only useful signal — the moment of divergence.

The corollary matters too: the rank that *reports* the timeout is a rank that
**arrived**. The culprit is one of the silent ones.

## Causes, ranked by base rate

Ordered roughly by how often each is the actual cause in practice. Work down
the list; do not start at the bottom.

**1. Divergent control flow.** Some ranks take a branch that calls a
collective and others do not. Classic sources:

- `if loss > threshold: continue` — skips a step on one rank only
- a validation or logging path guarded by `if rank == 0` that contains a
  collective, or that *skips* one
- unequal dataloader length: one rank exhausts its shard a step early and
  exits the loop while the others call all-reduce
- early `break` on a NaN check evaluated per rank
- dynamic batching where a rank gets zero samples

This is the most common cause by a wide margin, and it is also the easiest to
find: the divergence is deterministic and reproducible.

**2. Mismatched collective arguments.** Same call, different shape, dtype, or
group on different ranks. NCCL may hang rather than error, because it is
waiting for bytes that will never come in the amount it expects. Frequent
with dynamic shapes (variable sequence length, MoE routing counts).

**3. A rank died silently.** A segfault, a host OOM kill, or an exception
swallowed by a worker. The survivors then block forever on the next
collective. The timeout you see is a *symptom* of a crash that happened
earlier — check for a rank that stopped emitting logs before the hang.

**4. Mismatched collective ordering.** Rank A calls all-reduce then
all-gather; rank B calls them in the other order. NCCL matches operations by
their order within the group, so this deadlocks. Usually caused by
conditionally-executed modules.

**5. A genuine straggler.** One rank is slower for a physical reason —
thermal throttling, an ECC-degraded GPU, a slower NIC path, a noisy
neighbour, or simply longer input sequences. This does not time out at first;
it shows up as elevated step time and eventually as a timeout under load.

**6. Fabric or driver-level failure.** A link down, a NIC reset, GPUDirect
disabled. This is real, but it is *last* on the list because the first five
are far more common — and it is also the one that belongs to
`mlops:forge-train` to diagnose on a specific cluster.

## Localization procedure

The goal is to identify the rank that did not arrive and the operation it was
supposed to arrive at.

**Step 1 — read the timeout message properly.** The watchdog message names the
collective, its sequence number within the process group, the timeout
duration, and the rank that reported. The **sequence number** is the key: it
identifies *which* collective in the step failed, and comparing sequence
numbers across ranks tells you who is behind.

**Step 2 — use the flight recorder.** PyTorch maintains a ring buffer of
recent collective records per rank, enabled by setting a nonzero trace-buffer
size (`TORCH_NCCL_TRACE_BUFFER_SIZE`, named here to say what it does — the
value is a buffer depth, not a cluster setting). On timeout it dumps, per
rank, the recent collectives with their sequence numbers, shapes, and
start/completion state.

The dump answers the question directly:

```
ranks 0..6:  seq 4127  all_reduce  [started, not completed]
rank 7:      seq 4126  all_gather  [completed]   ← last thing rank 7 did
```

Rank 7 never reached collective 4127. Now look at what rank 7's code path
does between 4126 and 4127 — that is the divergent branch.

**Step 3 — if the flight recorder is unavailable, bisect by logging.** Log
`(rank, step, tag)` immediately before every collective. The rank whose log
stops first is the one to investigate. Crude, but it works and needs nothing
special.

**Step 4 — reproduce smaller.** Nearly every cause in classes 1–4 reproduces
on 2 ranks on one node. If it does not, you are in class 5 or 6.

**Step 5 — for arguments/order mismatches**, enable NCCL's own debug output
at a subsystem-scoped verbosity (`NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS` control
this) and compare the collective sequence across ranks. Watch out: high
verbosity changes timing enough to hide race-dependent hangs.

## Detecting a straggler before it becomes a timeout

Stragglers are the failure mode that does not announce itself. Instrument
for them proactively:

| Instrument | What it reveals |
|---|---|
| per-rank step time, p50 / p99, logged every step | one rank consistently above the rest |
| per-rank time spent *inside* collectives | fast ranks wait longer — the **inverse** signal |
| per-rank tokens processed per step | data skew (variable-length sequences) |
| SM clock and GPU temperature per rank | thermal throttling |
| per-rank host CPU time in the dataloader | a slow disk or a starved worker |

The inverse signal is the useful one and is often missed: **the straggler
spends the least time waiting in collectives**. If rank 5 reports 2 ms in
all-reduce while everyone else reports 80 ms, rank 5 is the straggler — the
others are waiting for it. Ranking by "time in NCCL" descending finds the
victims; ranking ascending finds the culprit.

Systematic causes worth checking, in order:

1. **Data skew.** Variable-length sequences, unsorted batches, or a shard with
   longer documents. Fix by bucketing or padding to a fixed length.
2. **Causal-mask imbalance under context parallelism** — per-rank time rising
   monotonically with rank index. See
   `parallelism-strategies/references/sequence-context-parallel.md`.
3. **MoE routing imbalance** — step time varying step to step rather than by
   rank. See `parallelism-strategies/references/expert-parallel-moe.md`.
4. **Pipeline stage imbalance** — regular, not random; see
   `parallelism-strategies/references/pipeline-parallel.md`.
5. **Hardware.** Thermals, a degraded link, a card with reduced clocks.

## The cost of a straggler at scale

A collective's duration is the **maximum** over `n` ranks. If per-rank
duration has mean `μ` and the tail is heavy, the expected maximum grows with
`n` — so a jitter source that is invisible at 8 GPUs is dominant at 8192. This
is the "serialization" budget in the router's frame, and it is why large runs
invest in per-rank telemetry that small runs never need.

## Checkpoint interval: Young/Daly

At scale, failures stop being exceptional (router →
`distributed-training-router/references/why-scaling-is-hard.md`): with per-GPU MTBF `MTBF_1`,
`MTBF_job ≈ MTBF_1/n`. So checkpointing frequency becomes an optimization.

Let `C` be the wall-clock cost of writing one checkpoint and `M` the job-level
MTBF. Checkpoint every `T` seconds. Per interval you pay:

```
overhead(T) = C            (the write itself)
            + T/2          (expected work lost, if a failure lands uniformly in the interval)
                × (T/M)    (probability a failure lands in this interval)
```

Fraction of time wasted:

```
f(T) = C/T + T/(2M)
```

Differentiate and set to zero:

```
df/dT = −C/T² + 1/(2M) = 0    →    T_opt = sqrt(2·C·M)
```

```
T_opt ≈ sqrt(2 · C · MTBF)
```

(Daly, "A higher order estimate of the optimum checkpoint interval for restart
dumps", *Future Generation Computer Systems* 22(3), 2006, pp. 303–312. Daly's
higher-order form corrects this first-order estimate when `C` is not small
relative to `M`; the square-root form is the one to reason with.)

Worked: `C = 120 s` to write a sharded checkpoint, `M = 3 hours = 10800 s`.

```
T_opt = sqrt(2 · 120 · 10800) = sqrt(2.59e6) ≈ 1610 s ≈ 27 minutes
f(T_opt) = 120/1610 + 1610/21600 = 7.5% + 7.5% = 15%
```

Two readings that people find counterintuitive:

- At the optimum, the two terms are **equal**. If your checkpoint-write time
  and your expected-lost-work are not roughly equal, your interval is wrong.
- **Halving `C` is worth more than it looks.** `T_opt ∝ sqrt(C)` but
  `f(T_opt) = sqrt(2C/M)`, so halving checkpoint cost reduces total overhead
  by 1/√2 ≈ 29%. Asynchronous and sharded checkpoint writes pay for
  themselves quickly at scale.

For calibration on `M`: Llama 3's 405B run recorded 466 interruptions over 54
days on up to 16K H100s — 419 unexpected, ~78% of those confirmed hardware,
GPU failures 58.7% of the unexpected total (arXiv:2407.21783). That is an
unexpected-failure MTBF of roughly 3 hours.

## Quick checklist for a hang

1. Which rank *reported*? It is not the culprit.
2. Is there any rank whose logs stopped earlier? → class 3, it crashed.
3. Does the flight-recorder dump show a sequence-number mismatch? → the rank
   with the lower number is the one that did not arrive.
4. What runs between those two sequence numbers? → look for a branch.
5. Do all ranks have the same number of batches? → dataloader length.
6. Are any collective shapes data-dependent? → class 2.
7. Only now consider the fabric — and hand that to `mlops:forge-train`.
