---
name: communication-backends
description: |
  Everything on the wire between GPUs and between nodes. Activate for: ring vs
  tree all-reduce, collective algorithms and the α-β cost model, all-gather /
  reduce-scatter / all-to-all, busbw vs algbw and nccl-tests, gradient bucketing,
  overlapping comm with compute, NVLink / PCIe / InfiniBand topology and rail
  alignment, why a collective timeout or hang happens ("Watchdog caught
  collective operation timeout", NCCL timeout), straggler detection,
  flight recorder, checkpoint-interval math, gradient compression,
  "NCCL all-reduce가 ring이랑 tree랑 뭐가 달라".
  Do NOT activate for how much memory a model needs, OOM or offload
  (memory-offloading), parallelism degrees or ZeRO stage
  (parallelism-strategies), single-GPU kernels or roofline (gpu-architecture),
  or MFU/throughput numbers (training-metrics). Do NOT
  activate for real cluster operations — a job's NCCL logs, NCCL_IB_HCA /
  NCCL_SOCKET_IFNAME values for a specific cluster, submit, quota, multi-node
  launch: mlops:forge-train.
---

# Communication Backends

The cost model, algorithms, and failure modes of everything that crosses a
link. This skill explains *mechanism*. It does not read your job's logs and it
does not tell you what to set on your cluster.

> Platform recipe: concrete NCCL environment values for a specific fabric,
> reading a failed job's NCCL output, and multi-node launch belong to skill
> `mlops:forge-train` → `error-patterns.md` §3 (NCCL). Naming a variable such
> as `NCCL_ALGO`, `NCCL_PROTO`, or `TORCH_NCCL_TRACE_BUFFER_SIZE` to explain
> what it *controls* is in scope here; telling the user what value to set for
> their cluster is not.

Symbols (`n`, `S`, `α`, `β`, `Ψ`, `b`, `s`, `h`) and the `busbw`/`algbw` and
Gb/s-vs-GB/s conventions are defined in
skill `distributed-train:distributed-training-router`, file
`distributed-training-router/references/notation-and-glossary.md`.

## The α-β model — the spine of this whole skill

Every collective's cost is modelled as a sum of a per-message latency and a
per-byte transfer term:

```
T(S) = α + β·S            α = per-message latency (s), β = 1/bandwidth (s/byte)
```

Sending `k` separate messages of size `s` costs `k(α + β·s)`; sending one
message of size `k·s` costs `α + β·k·s`. The difference, `(k−1)·α`, is the
entire justification for gradient bucketing, for FSDP wrapping granularity,
and for why interleaved pipeline schedules get expensive on high-latency
fabrics.

Three regimes follow, and naming the regime is the first step of any
communication diagnosis:

| Regime | Condition | Dominant term | Fix direction |
|---|---|---|---|
| **latency-bound** | `S ≪ α/β` | `α`, and the number of messages | fewer, larger messages; tree algorithms |
| **bandwidth-bound** | `S ≫ α/β` | `β·S` | fewer bytes: compression, lower precision, a different parallelism axis |
| **overlap-bound** | either, but compute is idle waiting | scheduling | prefetch, bucket boundaries, `no_sync()` |

The crossover `S* = α/β` is a property of the fabric. For `α = 5 µs` and
`β = 1/(25 GB/s)`, `S* = 125 KB` — messages smaller than that are latency
dominated. This is why 100 tiny gradient tensors are far worse than one
25 MB bucket.

## Ring all-reduce and its optimality

Derived from scratch in `references/collective-algorithms.md`. The result:

```
ring all-reduce time  =  2(n−1)·α  +  2(n−1)/n · S · β
```

The bandwidth term `2(n−1)S/n → 2S` is a **lower bound** for all-reduce — no
algorithm can do better, because every rank must both send its data out and
receive the reduced result. Ring achieves it. The latency term `2(n−1)α`
grows linearly in `n`, which is what breaks ring at large scale.

## The identity that makes ZeRO-1/2 free

```
AllReduce  =  ReduceScatter  ∘  AllGather
    2(n−1)/n · S   =   (n−1)/n · S   +   (n−1)/n · S
```

Ring all-reduce literally executes these two phases. Consequently:

- ZeRO-1 and ZeRO-2 replace one all-reduce with a reduce-scatter plus an
  all-gather at **identical total volume** — the `2Ψ` in
  `parallelism-strategies/references/data-parallel-and-zero.md`.
- Megatron sequence parallelism replaces TP's all-reduce with an
  all-gather/reduce-scatter pair at identical volume, buying activation memory
  for free.

Whenever you see "we replaced an all-reduce with RS+AG", the correct reaction
is *that is volume-neutral* — the gain is elsewhere.

## Double binary trees

Trees fix ring's latency problem: a broadcast down a binary tree takes
`log₂(n)` steps instead of `n−1`. But a plain binary tree wastes bandwidth —
leaves only receive, never forward.

NCCL's answer: build **two** complementary trees. In any binary tree, at
least half the ranks are leaves. Construct a second tree in which the leaves
of the first are internal nodes and vice versa, then send half the data down
each. Every rank now both sends and receives at full rate, and:

```
tree all-reduce latency ≈ 2·log₂(n)·α          (vs 2(n−1)·α for ring)
```

At `n = 1024`: ring pays `2046·α`, trees pay `20·α` — a 100× reduction in the
latency term, while retaining full bandwidth. NCCL selects between ring and
tree by message size and world size; trees win for small/medium messages,
rings for large ones where the bandwidth term dominates anyway.
(NVIDIA, "Massively Scale Your Deep Learning Training with NCCL 2.4":
https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4)

`NCCL_ALGO` and `NCCL_PROTO` are the variables that override this selection —
named here to explain the mechanism, not as a recommendation.

## Bucket size, derived

Gradients arrive during backward one tensor at a time, and a transformer has
thousands of small tensors. Firing a collective per tensor costs `k·α`.
Bucketing accumulates gradients until a byte threshold is reached, then fires
one collective.

```
per-tensor:   k·(α + β·s)   = k·α + β·S
one bucket:   α + β·k·s     = α   + β·S
saving:       (k−1)·α
```

But a bucket that is too large delays the *start* of communication until many
tensors are ready, shrinking the window available for overlap. The optimum
balances `α` amortization against overlap loss; PyTorch DDP's 25 MB default
sits comfortably above `S* = α/β` for typical fabrics while still allowing
several buckets per backward pass. The full argument, including why bucket
order must follow *reverse* forward order, is in
`references/overlap-and-scheduling.md`.

## Hangs and stragglers: the one insight

**A collective is a barrier.** `all_reduce` does not return on any rank until
every rank in the group has called it. Therefore:

```
A collective timeout does not mean the network is slow.
It means some rank never arrived at the collective.
```

Raising the timeout almost never fixes anything — it converts a 10-minute
failure into an hour-long one. The correct move is to find *which* rank did
not arrive and *what it was doing instead*. `references/hang-and-straggler-debugging.md`
gives causes ranked by base rate, the flight-recorder localization procedure,
and the Young/Daly checkpoint-interval result
`T_opt ≈ sqrt(2·C·MTBF)` for deciding how much work a crash should cost.

## Diagnosing a communication problem

1. **Name the regime.** Message size vs `S* = α/β`. Latency, bandwidth, or
   overlap.
2. **Compare against the wire, in `busbw`.** `algbw` never reaches link
   speed and comparing it to the datasheet produces false alarms.
   `busbw = algbw · 2(n−1)/n` for all-reduce.
3. **Check the boundary.** If efficiency is fine within a node and falls
   across nodes, the parallelism *ordering* is wrong before the fabric is.
   Route to `parallelism-strategies`.
4. **Check variance before blaming bandwidth.** Stable slow → bandwidth.
   Variable slow → straggler or bubble. This is the router's separating test.
5. **Confirm overlap actually happened.** A profile showing NCCL kernels
   serialized after backward means the overlap conditions were violated —
   usually gradient accumulation without `no_sync()`, or a `.item()` sync in
   the loop.

## Rules for answering here

- Always state which term of `α + β·S` you are addressing. "It's the network"
  is not a diagnosis.
- Use `busbw` when comparing to hardware, `algbw` when reporting application
  throughput, and say which.
- Convert vendor Gb/s to GB/s explicitly.
- For a timeout, name the rank-identification procedure before any
  configuration change.
- Do not emit `forge` commands or cluster-specific env-var values. Naming a
  variable to explain a mechanism is fine.

## Reference files

| File | Contents |
|---|---|
| `references/collective-algorithms.md` | ring all-reduce derived step by step, the `2(n−1)S/n` bound, RS∘AG, trees, all-to-all, cost table for every collective |
| `references/bandwidth-and-topology.md` | the bandwidth hierarchy, NVLink/PCIe/IB, rail alignment, oversubscription, hierarchical collectives, busbw arithmetic |
| `references/overlap-and-scheduling.md` | bucket-size derivation, backward-overlap conditions, FSDP prefetch, what breaks overlap, reading overlap in a trace |
| `references/hang-and-straggler-debugging.md` | the barrier insight, causes by base rate, flight recorder, straggler detection, Young/Daly checkpoint interval |
| `references/compression-and-quantized-collectives.md` | where compression can and cannot help, error feedback, 1-bit Adam/LAMB, PowerSGD, fp8/int8 collectives, when the answer is "no" |
