# Allocator, Fragmentation, and the OOM Playbook

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`.

> Platform recipe: if the OOM appears in a **real job's logs**, or the process
> was killed by the host (exit 137 is a host-side kill, not a CUDA OOM), that
> is skill `mlops:forge-train`. This file explains the memory model and the
> fixes.

## How the caching allocator works

`cudaMalloc` is slow and synchronizing, so PyTorch does not call it per
tensor. Instead:

1. It requests large **segments** from CUDA and keeps them.
2. Tensor allocations are served by splitting **blocks** out of segments.
3. Freed blocks return to a pool, not to CUDA.
4. Only under memory pressure does it release empty segments back.

Hence the two numbers in every OOM message:

```
allocated  = bytes in live tensors
reserved   = bytes the allocator holds from CUDA (allocated + cached free blocks)
```

```
reserved − allocated = memory you paid for and cannot currently use
```

The allocator keeps separate pools for small (<1 MB) and large allocations to
limit how badly the two interfere.

## The fragmentation signature

```
CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 79.15 GiB of which 1.28 GiB is free.
Of the allocated memory 61.20 GiB is allocated by PyTorch, and
12.44 GiB is reserved by PyTorch but unallocated.
```

12.44 GB reserved-but-unallocated while a 2 GB request fails. There *is* free
memory; there is no free **contiguous block** of 2 GB. That is fragmentation,
not a shortage, and the fixes are different.

Causes, in order of frequency:

| Cause | Mechanism |
|---|---|
| Varying tensor shapes | variable sequence length; each new shape needs a differently-sized block |
| Mixed allocation sizes | large and small allocations interleaved, splitting segments |
| Long-lived allocations interleaved with short-lived | a permanent block in the middle of a segment prevents coalescing |
| Repeated grow/shrink cycles | eval with a different batch size, then back to train |
| A memory spike early in the run | sets the segment layout for everything after |

## Fixes for fragmentation

**1. `expandable_segments:True`.** The highest-value single change. Set via
`PYTORCH_CUDA_ALLOC_CONF`, it lets the allocator grow an existing segment via
virtual-memory mapping instead of allocating a new fixed-size one. Most
shape-variation fragmentation disappears. It is the first thing to try when
`reserved − allocated` is large.

**2. Make shapes uniform.** Pad sequences to a fixed length, or bucket lengths
into a small number of sizes. This also helps `torch.compile` (fewer
recompilations) and tile alignment
(`gpu-architecture/references/tensor-cores-and-precision.md`) — three benefits
from one change.

**3. `torch.cuda.empty_cache()` — sparingly.** It releases cached segments back
to CUDA. It is synchronizing and slow, so it does not belong in the training
loop. It is legitimate once, between phases with very different allocation
patterns (e.g. after loading a checkpoint, or between train and a
differently-shaped eval).

**4. Allocate the big things first.** A large allocation made when memory is
mostly free gets a clean segment. The same allocation made later may not fit
anywhere.

**5. Do not allocate in the loop what you can allocate once.** Reusing a
preallocated buffer avoids the whole cycle.

## Reading a memory snapshot

PyTorch can record every allocation with its stack trace:

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run a few steps, ideally through the point of failure ...
torch.cuda.memory._dump_snapshot("snap.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

The dump renders as a timeline of allocations with sizes and stacks. What to
look for:

| Pattern | Meaning |
|---|---|
| A large flat band across the whole run | model states — expected |
| A sawtooth rising through forward and falling through backward | activations — expected |
| A peak at the optimizer step | optimizer state + fp32 master |
| A band that grows step over step | **a leak** — something is retaining tensors |
| Many small blocks scattered between large ones | fragmentation |
| A single very tall spike | one allocation is your peak; find it and shrink it |

The peak is what matters, not the average. A run that sits at 40 GB and spikes
to 79 GB for 5 ms OOMs exactly as reliably as one that sits at 79 GB.

## Memory leaks in training loops

"OOM after N successful steps" with a constant workload is almost always a
retained reference rather than fragmentation. The usual suspects:

| Cause | Fix |
|---|---|
| `losses.append(loss)` — retains the whole autograd graph | append `loss.item()` or `loss.detach()` |
| Accumulating metrics as tensors | `.item()` them |
| A closure or hook capturing an activation | check custom hooks |
| `loss.backward(retain_graph=True)` where it is not needed | remove it |
| Keeping the previous step's output for logging | detach it |
| An exception handler holding a traceback that references tensors | clear it |

The `.append(loss)` case is the single most common bug of this class: each
retained loss tensor keeps its entire graph, so memory grows by roughly one
step's activations per step.

## The full OOM playbook

**Step 1 — classify.** Where in the step did it fire?

| Fires at | Term | Go to |
|---|---|---|
| model init / before step 1 | model states | step 3 |
| forward | activations | step 4 |
| start of backward | peak activations + grads | step 4 |
| `optimizer.step()` | optimizer state | step 5 |
| eval | eval config | check `no_grad()` and eval batch size |
| after N steps | fragmentation or leak | step 2 |

**Step 2 — fragmentation or leak?**
`reserved − allocated` large → fragmentation → `expandable_segments:True`,
uniform shapes.
`allocated` itself growing step over step → leak → the table above.

**Step 3 — model states do not fit.** Batch size is irrelevant here.
`16Ψ` (or your `K`) vs capacity. Options in order: ZeRO-1 → ZeRO-2 →
bf16 moments → 8-bit optimizer → ZeRO-3 → TP → CPU offload.
(`references/optimizer-and-precision-memory.md`,
`parallelism-strategies/references/data-parallel-and-zero.md`.)

**Step 4 — activations do not fit.** In priced order:

```
1. lower b, raise m to hold B          ~free (and often improves the comm ratio)
2. fused attention kernel               negative cost — it is faster
3. selective recomputation              ~2-5%
4. sequence parallelism (with TP)       ~0% extra comm
5. full recomputation                   ~25-33%
6. TP (+SP) / CP                        comm, needs the right interconnect
7. PP                                   bubble
```

**Step 5 — optimizer state does not fit.** ZeRO-1 first (free), then bf16
moments, then 8-bit, then CPU offload
(`references/offload-to-cpu-and-nvme.md`).

**Step 6 — re-measure.** Every one of these changes the step time or the
communication volume. Confirm the fix did not cost more than it was worth:
`training-metrics/references/benchmarking-methodology.md`.

## Things that do not help (and why people try them)

| Attempted fix | Why it does not work |
|---|---|
| Adding more GPUs under **DDP** | DDP replicates model states; per-GPU memory is unchanged by `n` |
| Lowering `b` when model states are the problem | states are independent of batch size |
| `empty_cache()` in the training loop | releases cached blocks that were about to be reused; slow and synchronizing |
| Raising the NCCL timeout | that is a hang, not an OOM |
| `del` on a tensor that is still referenced elsewhere | the reference count is what matters |
| Switching fp16 → bf16 to save memory | same 2 bytes; it fixes range, not capacity |
| Gradient checkpointing when the problem is model states | it only touches activations |

The first row is worth stating explicitly whenever someone proposes scaling
out to fix an OOM: **under DDP, more GPUs is not more memory.** Sharding is
what converts world size into capacity.

## Quick reference: the numbers

```
model states       = (2 + 2 + K)·Ψ bytes            K = 12 for fp32-master Adam
                     ÷ d under ZeRO-3
activations        = L · s·b·h·(34 + 5as/h)         ÷ t with SP, ÷ c with CP
                     → 34·s·b·h·L/t with selective recompute
framework overhead ≈ 2–4 GB
headroom           ≈ 10% for fragmentation
```
