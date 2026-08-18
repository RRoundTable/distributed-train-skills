# Overlap and Scheduling

Communication that hides behind compute is free. Communication that does not
is pure cost. This file is about the difference. Symbols: router →
`distributed-training-router/references/notation-and-glossary.md`.

## Why overlap is possible at all

The backward pass computes gradients layer by layer, from the output back to
the input. The gradient for the last layer is final long before the first
layer's is even started. Since the all-reduce for layer `L` depends only on
layer `L`'s gradient, it can run **concurrently** with the backward compute of
layers `L−1, L−2, …`.

GPUs support this because collectives run on their own CUDA streams and NCCL
kernels occupy a small number of SMs, leaving the rest for compute. The
theoretical best case:

```
T_step = max(T_compute, T_comm) + T_unhidden_tail
```

instead of `T_compute + T_comm`. The tail is the last bucket's all-reduce,
which has no remaining backward compute to hide behind — irreducible, and the
reason overlap gets you to ~90%, not 100%.

## Bucket size, derived

Gradients arrive as thousands of small tensors. Two extremes:

```
one collective per tensor (k tensors, s bytes each):
    T = k(α + β·s) = k·α + β·S

one collective for everything:
    T = α + β·S
```

The saving is `(k−1)·α`, which for `k = 2000` tensors and `α = 5 µs` is
10 ms per step — significant. But the single-bucket extreme cannot start
until the *last* gradient is ready, so it overlaps with nothing.

The real objective is to minimize the unhidden portion:

```
minimize   T_unhidden = Σ_buckets max(0, T_comm(bucket_i) − T_compute_available_after(bucket_i))
```

The structure of the optimum:

- Bucket size should comfortably exceed `S* = α/β` so each collective is in
  the bandwidth regime. For `α = 5 µs`, `β = 1/25 GB/s`, `S* = 125 KB`.
- There should be enough buckets that the first fires early in the backward
  pass. Typically 4–20 buckets per step.
- PyTorch DDP's 25 MB default sits ~200× above `S*` and yields a handful of
  buckets for a multi-billion-parameter model — a reasonable point, though
  worth sweeping for unusual model shapes or fabrics.

## Bucket ordering: reverse of forward

DDP assigns tensors to buckets in the **reverse** order of their appearance in
the forward pass, because that approximates the order gradients become ready
in backward. If the order is wrong, a bucket waits on a gradient that arrives
last, delaying a collective that could have started much earlier.

This is also why `find_unused_parameters=True` costs so much: DDP must
traverse the autograd graph to discover which parameters received gradients
before it can decide a bucket is complete, serializing what should be a
streaming process. Use it only when the model genuinely has conditionally-used
parameters, and prefer restructuring the model so it does not.

## What breaks overlap

| Cause | Mechanism | Signature in a trace |
|---|---|---|
| Gradient accumulation without `no_sync()` | an all-reduce fires every micro-batch instead of every step | `m×` the expected NCCL time |
| A host sync in the loop (`.item()`, `.cpu()`, `print(loss)`) | CPU waits for the GPU, so no further kernels are queued | a long gap with no kernels at all |
| `find_unused_parameters=True` | graph traversal before bucket completion | NCCL starts late in backward |
| Too-large buckets | first collective cannot start until many gradients are ready | NCCL clustered at the end of backward |
| Too-small buckets | `α` dominates | many short NCCL kernels, low `busbw` |
| Gradient clipping on the full flat gradient before the reduce | a global norm needs *all* gradients — a hard barrier | NCCL entirely after backward |
| CPU-bound launch (small kernels, Python overhead) | there is no compute to hide behind | gaps everywhere, low GPU util |
| A `.grad` access or optimizer hook mid-backward | forces synchronization | irregular gaps |

The gradient-clipping row surprises people. Computing a **global** gradient
norm requires every gradient, so `clip_grad_norm_` naturally serializes after
the last all-reduce. Frameworks avoid the worst of it by computing partial
norms per bucket and reducing the scalars, which is cheap — but a hand-rolled
clip over `model.parameters()` reinstates the barrier.

## FSDP / ZeRO-3 prefetch

ZeRO-3 adds a second overlap problem: parameters must be *gathered* before a
layer can run.

```
forward:   AG(layer i+1)  ‖  compute(layer i)
backward:  AG(layer i−1)  ‖  compute(layer i)   +  RS(grad i)  ‖  compute(layer i−1)
```

Both frameworks prefetch the next unit's all-gather while the current unit
computes. Two knobs decide whether this works:

- **Wrapping granularity.** One transformer block per unit is the standard
  answer: large enough that the all-gather is in the bandwidth regime, small
  enough that the gather for block `i+1` fits inside block `i`'s compute. A
  whole-model wrap gives one enormous gather with nothing to hide behind; a
  per-parameter wrap gives thousands of latency-bound gathers.
- **Prefetch depth.** Gathering `i+1` while computing `i` needs both units'
  parameters resident simultaneously — a memory-for-overlap trade. Deeper
  prefetch overlaps more and costs more HBM.

The backward pass is the harder case: the first backward unit needs its
parameters immediately, with no preceding compute to hide the gather, and
`reshard_after_forward=False` on the outermost layers is the standard partial
fix (they are the first to be needed again).

## Compute/communication balance

Overlap only helps if there is compute to hide behind. From the router's
compute-to-communication ratio:

```
ratio = 6 · b · m · s / bytes_per_elem     [FLOPs per byte on the wire]
```

If `T_comm > T_compute`, no scheduling fixes it — the collective is on the
critical path by arithmetic. The levers are then:

| Lever | Effect on ratio | Cost |
|---|---|---|
| raise `m` (gradient accumulation) | linear increase | fewer optimizer steps per unit time at fixed `B`, or larger `B` |
| raise `b` | linear increase | memory |
| raise `s` | linear increase | changes the experiment |
| bf16 instead of fp32 gradients | 2× | usually already done |
| ZeRO-2 instead of ZeRO-3 | 1.5× | memory |
| compression | up to 32× on the wire | convergence risk — see the compression file |

## Reading overlap in a profile

What you want to see: NCCL kernels running **concurrently** with compute
kernels on the timeline, starting early in the backward pass, with only the
final bucket exposed at the end.

| Pattern | Meaning |
|---|---|
| NCCL kernels interleaved with backward compute | overlap working |
| One long NCCL block after all backward compute | overlap broken — check the table above |
| NCCL blocks between every micro-batch | missing `no_sync()` |
| Compute and NCCL both idle, long gaps | CPU-bound or a host sync |
| NCCL kernels present but very short and numerous | buckets too small |
| Regular idle at fixed offsets in the step | pipeline bubble, not a comm problem — `parallelism-strategies` |

> Platform recipe: capturing a profile from a job running on a cluster, and
> retrieving the trace file, is skill `mlops:forge-train`. Interpreting the
> single-GPU kernel side of it is `distributed-train:gpu-architecture` →
> `gpu-architecture/references/profiling-with-nsight.md`.

## Checklist

1. Is there a host sync inside the training loop? Remove it first — it makes
   every other measurement lie.
2. With gradient accumulation, is `no_sync()` (or the framework equivalent)
   in use?
3. Is bucket size above `S* = α/β` for this fabric, and are there enough
   buckets to start early?
4. Under FSDP, is the wrapping unit one transformer block?
5. Is `T_comm` actually smaller than `T_compute`? If not, stop tuning the
   schedule and change the ratio.
