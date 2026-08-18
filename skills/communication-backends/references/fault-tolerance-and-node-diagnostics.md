# Fault Tolerance and Node Diagnostics

> **Deferral banner.** Evicting a node, reading a real job's health metrics,
> restarting a real job, and every cluster-specific tool name or threshold
> belong to skill `mlops:forge-train`. This file explains *why* the loop has
> the shape it has and *which test isolates which scope*, so that the tooling
> you use there is pointed at the right thing. It contains no cluster-specific
> values.

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`.

## Goodput: the metric MFU alone cannot express

MFU measures how well a step ran. It says nothing about how many steps ran at
all. At scale the two are separate problems, and the product is what the run
actually delivers:

```
goodput  =  MFU  ×  ETTR

ETTR (effective training time rate)
   = time spent in useful steps / total wall-clock time of the run
   (excludes: crash-to-detection, diagnosis, restart, re-doing work lost
    since the last checkpoint)
```

A run at 55% MFU and 70% ETTR loses more to availability than a run at 45%
MFU and 95% ETTR loses to efficiency. MegaScale reports **ETTR above 90%**
across a multi-week run on more than 10,000 GPUs, with **over 100 automatic
recoveries** during it (arXiv:2402.15627). Llama 3's 405B run recorded **466
interruptions in 54 days** (arXiv:2407.21783). Neither number is an anomaly;
it is what `MTBF_job ≈ MTBF_1/n` predicts
(`distributed-training-router/references/why-scaling-is-hard.md` §6).

The practical consequence: past a few hundred GPUs, an optimization that adds
2 points of MFU and 1 point of restart overhead is a **loss**.

## The recovery loop

Every production system converges on the same four phases, because each one
answers a question the next one needs:

```
detect   →   diagnose   →   evict & replace   →   resume
 "is something wrong?"  "which node?"  "swap it"  "from where?"
```

| Phase | Signal it runs on | Failure mode if skipped |
|---|---|---|
| **Detect** | heartbeat from a per-node daemon; process exit status; log scraping; step-time and traffic anomalies | the job hangs until a collective timeout — tens of minutes of dead GPUs |
| **Diagnose** | a bounded self-check suite (below), run on all nodes with training suspended | you evict a healthy node and hit the same failure again |
| **Evict & replace** | scheduler API; blocklist of bad node addresses | the same bad node is rescheduled into the next attempt |
| **Resume** | the last checkpoint | correct but slow — this is where checkpoint cost is paid |

The **detect** phase is where most homegrown setups are weakest. Waiting for
the NCCL watchdog means waiting for the timeout, which is deliberately long.
Cheaper signals that fire first:

- a per-node daemon heartbeat (absence is unambiguous, unlike a slow step)
- stdout/stderr scraped for known error and warning strings as they appear
- **RDMA/NIC traffic volume per step.** Training is periodic, so per-step
  traffic should be nearly identical step to step. A sharp decline or an
  unusual fluctuation is an early warning that costs nothing to watch.

That last one generalizes past networking: **anything periodic in a training
step is a free anomaly detector.** Step time, bytes on the wire, and tokens
consumed all have this property.

## The self-check suite, scoped by level

The design constraint is that this runs on *every* node with training
suspended, so it must be fast and it must be diagnostic — a test whose failure
does not localize is worse than no test. Order the tests so that each adds
exactly one level of scope:

| Level | Test | What its failure isolates |
|---|---|---|
| Host internals | loopback bandwidth from each NIC to intra-host endpoints | a bad PCIe path, a wrong NUMA/topology binding, a degraded slot |
| Host internals | NIC-to-NIC bandwidth within the same host | one bad card, distinguished from a bad path |
| Intra-node GPU | all-to-all across the GPUs of one node | NVLink/NVSwitch degradation, a card with reduced clocks |
| Nearest neighbours | all-reduce between adjacent hosts under the same leaf switch | the cable, the transceiver, the leaf port — without involving the spine |

Each level involves everything the previous one did plus one new component, so
the first failing level names the component. Running only the last test — a
full-cluster all-reduce benchmark — tells you the cluster is slow and nothing
more.

Two properties worth copying:

- **The suite is bounded.** It is a health check, not a benchmark sweep. It
  runs in the time you are willing to spend with the whole job stopped.
- **There is a manual blocklist.** Some faults are only found by a human
  reading a trace days later. A path to mark a node bad without a
  reproducing test prevents the same node from poisoning run after run.

## Detecting the fault when nothing errors

Two cases defeat the loop above, and both are covered in
`references/hang-and-straggler-debugging.md`:

- **The culprit is silent.** A defective GPU that blocks a collective produces
  timeout logs on every rank *except* the one at fault. Ranking by "who
  reported an error" finds victims. See the barrier insight in that file.
- **The straggler never errors at all.** It just makes every collective wait.
  Detection is per-rank telemetry, not error handling.

Localizing either one requires mapping a rank index onto its position in the
`d·t·p·c` mesh. Under 3D or 4D parallelism the neighbourhood that matters is
not the numeric neighbourhood: rank 512 may be a TP peer of rank 513 and a DP
peer of rank 4608. A view that renders rank → mesh position, and the
collectives in flight on each, converts "some collective timed out" into "this
node, in this TP group". Build the mapping before you need it; deriving it
during an outage is where the hours go.

## Making the checkpoint cheap instead of rare

Young/Daly (`references/hang-and-straggler-debugging.md`) gives
`T_opt ≈ sqrt(2·C·MTBF)` and total overhead `f(T_opt) = sqrt(2C/M)`. Both are
governed by `C`, the wall-clock cost of one checkpoint — so the highest-value
move is not tuning the interval, it is **attacking `C`**.

**Two-stage saving.** Split the write at the memory boundary:

```
stage 1 (blocking):  GPU HBM  →  pinned host memory      seconds, PCIe-bound
stage 2 (async):     host memory  →  distributed storage  minutes, off the critical path
```

Training resumes after stage 1. The blocking cost is now a PCIe copy, not a
filesystem write, and `C` drops by an order of magnitude. The exposure this
buys back is a window in which a host failure loses the in-flight copy — a
much rarer event than the failures the checkpoint exists for.

**Recovery is the other half, and it is a different bottleneck.** On restart,
every rank reading its shard from shared storage means `n` concurrent readers
against one filesystem. But ranks in the same data-parallel group need the
*same* state. So: one rank per DP group reads from storage, then broadcasts
over the fabric — which is fast and, unlike the filesystem, scales.

```
naive:      n readers × shard bytes   from storage
broadcast:  n/d readers               from storage, remainder over NVLink/IB
```

**What this does to the arithmetic.** Take the worked example from the
Young/Daly section, `C = 120 s` and `M = 3 h`:

```
C = 120 s:   T_opt ≈ 1610 s (27 min),   overhead = 15.0%
C =   5 s:   T_opt ≈  329 s ( 5.5 min), overhead =  3.0%
```

Same MTBF, same everything else — 12 points of goodput, purely from making the
write cheap. And the shorter interval means less work lost per failure, which
is the part users actually feel.

## Non-steady-state costs do not appear in the step-time model

`T = T_compute/n_eff + T_comm(n) + T_bubble(p) + T_straggler` models a step.
Restarting is not a step, and at scale the things outside the step can
dominate the calendar:

- **Process-group initialization.** A naive rendezvous through a
  single-threaded key-value store, plus a global barrier implemented as
  `O(n²)` pairwise checks, is invisible at 64 GPUs and fatal at thousands.
  MegaScale reports NCCL group initialization taking **1,047 s at 2,048 GPUs**,
  reduced to **under 5 s** by replacing the store with one that handles
  concurrent clients and reducing the barrier to `O(n)`.
- **Checkpoint load**, addressed above.
- **Diagnosis**, which the self-check suite bounds.

17 minutes per attempt is a debugging tax paid on every restart, and it shows
up nowhere in an MFU number. When a job restarts a hundred times, these are
first-class costs. Measure them.

## Checklist for a run that must survive its own scale

1. Is there a signal that fires **before** the collective watchdog — a
   heartbeat, a log scrape, a per-step traffic check?
2. Does a failure produce a *node identity*, or only a rank number and a
   timeout? Is the rank → mesh-position map available offline?
3. Is there a bounded self-check suite, ordered so the first failure localizes?
4. Is `C` a filesystem write or a PCIe copy? Measure it — it sets both the
   interval and the total overhead.
5. On restart, how many ranks read from shared storage at once?
6. How long does the job take to get from launch to step 1? Is that number
   tracked across the run's lifetime?
7. Is ETTR reported alongside MFU? Without it, "goodput" is a guess.

## Sources

- MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs — arXiv:2402.15627
- The Llama 3 Herd of Models (interruption statistics) — arXiv:2407.21783
- Daly, *A higher order estimate of the optimum checkpoint interval for restart
  dumps*, Future Generation Computer Systems 22(3), 2006, 303–312
