---
name: distributed-training-router
description: |
  Entry point for distributed / large-scale model training questions that are
  vague, span several topics, or are a "where do I even start" ask. Activate for:
  "분산학습 뭐부터 봐야 해", "how do I train a 70B model", "what's the difference
  between DDP and FSDP and Megatron", "our training is slow, no idea why",
  "explain the whole distributed training stack", "which parallelism should I
  use", 처음 접하는 사람을 위한 개념 정리, or any question mixing memory +
  communication + throughput at once. Routes to the right sibling skill and
  supplies the shared notation, the four-budget diagnostic frame, and a
  symptom index.
  Do NOT activate for: questions already scoped to one surface — sharding and
  parallelism degrees (use distributed-train:parallelism-strategies), NCCL /
  collectives / interconnect (use distributed-train:communication-backends),
  SM/HBM/roofline/precision inside one GPU (use distributed-train:gpu-architecture),
  MFU / throughput / loss curves (use distributed-train:training-metrics),
  OOM, memory budgeting, "will this model fit on this GPU", activation
  checkpointing or offload (use distributed-train:memory-offloading).
  Also do NOT activate for running or inspecting real cluster jobs — submitting
  a training job, GPU quota, job status/QUEUED, reading a failed job's logs,
  building training images, datasets/disks, multi-node launch — that is
  mlops:forge-train.
---

# Distributed Training Router

Concept and theory layer for distributed training. This skill does **not**
run, submit, or inspect any job. It explains mechanisms, derives the math,
and points at the sibling skill that owns each surface.

> Platform recipe: anything that touches a real cluster — submitting a job,
> quota, images, disks, launcher flags, reading job logs, exit codes — belongs
> to skill `mlops:forge-train`. Hand off rather than guessing at CLI syntax.

## Start here

1. Read `references/notation-and-glossary.md` before quoting any formula.
   It fixes every symbol (`Ψ`, `n`, `L`, `h`, `a`, `s`, `b`, `m`, `B`, `V`,
   `d`, `t`, `p`, `c`, `e`, `v`, `K`, `α`, `β`), the mesh invariant
   `n = d·t·p·c`, the batch invariant `B = b·m·d`, bytes-per-element, and the
   Gb/s-vs-GB/s and `busbw`-vs-`algbw` traps.
2. Classify the question with the four-budget frame below.
3. Route to the owning skill. Do not answer a scoped question here — load the
   sibling and use its references.

## The four-budget frame

Every distributed training problem is a shortage of one of four things. They
have different fixes, and applying the wrong fix is the usual failure mode.

| Budget | The resource | Runs out as |
|---|---|---|
| **Compute** | FLOP/s the SMs can retire | slow steps, high utilization, high MFU already |
| **Capacity** | bytes resident in HBM | `CUDA out of memory` |
| **Bandwidth** | bytes/s across NVLink, PCIe, or the fabric | slow steps, low MFU, GPUs idle in collectives |
| **Serialization** | work that cannot overlap: bubbles, barriers, stragglers | slow steps, low MFU, *variance* across steps |

### The separating test

Three cheap observations pick the budget. Run them before proposing any fix.

```
OOM at all?                              → CAPACITY   → memory-offloading
MFU low AND GPU util high?               → COMPUTE    → gpu-architecture
MFU low AND GPU util low AND no OOM?     → BANDWIDTH or SERIALIZATION
   step time stable across steps?        →   BANDWIDTH → communication-backends
   step time varies run-to-run/rank?     →   SERIALIZATION → communication-backends
                                                (straggler / bubble)
MFU already high, want more?             → COMPUTE   → gpu-architecture
Don't know MFU yet?                      → measure   → training-metrics
```

Two caveats that make this test honest:

- **"GPU util" is nearly useless on its own.** `nvidia-smi` utilization is the
  fraction of time *any* kernel was resident, not the fraction of peak FLOPs
  achieved. A kernel spinning on a memory-bound copy reports 100%. Treat util
  as a *liveness* signal only; MFU is the real number. See
  `gpu-architecture/references/roofline-and-arithmetic-intensity.md`.
- **Capacity and bandwidth trade against each other.** Every technique that
  frees HBM buys it with FLOPs or with wire traffic. That trade is priced
  explicitly in `memory-offloading/references/allocator-fragmentation-and-oom-playbook.md`.

Read `references/why-scaling-is-hard.md` for how these four budgets interact
as `n` grows, and why the strong-scaling ceiling is set by the ratio of a
model's compute to its communication, not by any single component.

## Routing table

| The question is about | Load skill |
|---|---|
| Which parallelism, what degrees, ZeRO stage, sharding vs replication, TP/PP/CP/EP mechanics, Megatron layout, 3D/4D mesh design | `distributed-train:parallelism-strategies` |
| Anything on the wire: all-reduce/all-gather algorithms, ring vs tree, NCCL mechanics, bucket sizes, overlap, hangs, timeouts, stragglers, topology, compression | `distributed-train:communication-backends` |
| Inside one GPU: SMs, HBM/L2/SMEM, roofline, arithmetic intensity, tensor cores, bf16/fp8, kernel fusion, FlashAttention, Nsight profiling | `distributed-train:gpu-architecture` |
| Numbers you measured: FLOP counting, MFU/HFU, tokens/s, scaling efficiency, loss curves and spikes, scaling laws, benchmarking method | `distributed-train:training-metrics` |
| Memory capacity: what occupies HBM, OOM triage, activation recomputation, CPU/NVMe offload, optimizer/precision memory, fragmentation | `distributed-train:memory-offloading` |
| Vague, multi-topic, "where do I start", or a full-stack explanation | stay here |
| **A real job on a real cluster** | `mlops:forge-train` |

## Symptom index

`references/symptom-index.md` maps concrete observations — error strings,
curve shapes, profiler patterns, throughput numbers — onto the budget and the
owning skill. Use it when the user reports *what they saw* rather than asking
a conceptual question.

The most common entries:

| Observation | Budget | Skill |
|---|---|---|
| `torch.cuda.OutOfMemoryError` | capacity | memory-offloading |
| `Watchdog caught collective operation timeout` | serialization | communication-backends |
| 8 GPUs gave 5.2× speedup | bandwidth | training-metrics → communication-backends |
| MFU 18% on H100 with a 7B model | compute or bandwidth | training-metrics first |
| loss spiked at step 40k and never recovered | — | training-metrics |
| loss differs after changing TP degree | correctness | training-metrics (batch invariant) |
| step time p99 ≫ p50 | serialization | communication-backends |
| `reserved memory >> allocated memory` | capacity | memory-offloading |
| works on 1 node, hangs on 2 | serialization | communication-backends |

## Answering a "where do I start" question

For a newcomer, the useful order is capacity → parallelism → communication →
measurement, because each step's answer constrains the next:

1. **Will it fit?** `memory-offloading/references/memory-taxonomy-and-math.md`
   gives `HBM = model_states + activations + fragmentation + overhead`. If a
   single GPU holds the model states, everything below is optional.
2. **If not, what do you shard?** `parallelism-strategies` — ZeRO/FSDP first
   (it is pure sharding of things you already had), then TP inside the node,
   then PP across nodes.
3. **What does that cost on the wire?** `communication-backends` — the
   comm-volume table tells you whether the choice in step 2 fits the fabric.
4. **Did it work?** `training-metrics` — MFU and scaling efficiency, measured
   the same way twice, before and after.

Steps 1–3 are predictions. Step 4 is the only evidence. Never present a
configuration recommendation as a result; present it as a prediction with the
measurement that would falsify it.

## Rules for answering here

- Quote symbols only after checking them against
  `references/notation-and-glossary.md`. Silent symbol drift between files is
  the failure mode this glossary exists to prevent.
- State which budget you concluded, and which observation ruled the other
  three out. "Try gradient checkpointing" without naming the budget is a
  guess.
- Give numbers with their assumptions attached (`s`, `b`, `h`, dtype). A
  memory or bandwidth figure without them is not checkable.
- When a fix has a cost, price it. Every capacity fix costs compute or wire
  time; the tables in the sibling skills carry those prices.
- Do not emit `forge` commands, cluster-specific env-var *values*, or launcher
  recipes. Naming an env var to explain a mechanism is fine; telling the user
  what to set it to on their cluster is `mlops:forge-train`'s job.

## Reference files

| File | Contents |
|---|---|
| `references/notation-and-glossary.md` | the symbol table, both invariants, units hygiene, glossary, and where each formula is derived |
| `references/why-scaling-is-hard.md` | Amdahl and the strong-scaling ceiling, the compute-to-communication ratio, failure rates at scale, why the four budgets shift as `n` grows |
| `references/symptom-index.md` | observation → budget → skill, with the discriminating check for each |
