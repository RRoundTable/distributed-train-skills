---
name: memory-offloading
description: |
  What occupies GPU memory, why it runs out, and what to do about it.
  Activate for: CUDA out of memory triage and the order to try fixes,
  "A100 80GB에 13B bf16 + Adam 학습 가능해?", will this model fit, memory
  budgeting and the model-state vs activation split, activation memory formulas,
  gradient/activation checkpointing and selective recomputation, "gradient
  checkpointing 켜면 얼마나 느려져", CPU and NVMe offload, ZeRO-Offload and
  ZeRO-Infinity, optimizer state size, 8-bit optimizers, fp32 master weights as
  a memory line item, allocator fragmentation, "reserved but unallocated",
  expandable_segments, memory snapshots, and OOM that appears only after N steps.
  Do NOT activate for: choosing parallelism degrees or ZeRO stage as a sharding
  decision (use distributed-train:parallelism-strategies); collectives, NCCL,
  bandwidth or hangs (use distributed-train:communication-backends); roofline,
  kernel speed or precision numerics (use distributed-train:gpu-architecture);
  MFU, throughput or loss curves (use distributed-train:training-metrics).
  Also do NOT activate for real cluster operations — an OOM in a running or
  failed job's logs, exit 137 / host OOM kills, job submission, GPU quota,
  picking a GPU type for a job, or disks and datasets. That is mlops:forge-train.
---

# Memory and Offloading

The capacity budget: what is resident in HBM, why it does not fit, and the
priced ladder of things to do about it.

> Platform recipe: if the OOM is in a **real job's logs**, or the process was
> killed by the host (exit 137), or the question is which GPU type to request,
> that is skill `mlops:forge-train`. This skill explains the memory model and
> the fixes; it does not read job state.

Symbols (`Ψ`, `K`, `L`, `h`, `a`, `s`, `b`, `m`, `d`, `t`, `p`, `c`) and
bytes-per-element are defined in
skill `distributed-train:distributed-training-router`, file
`distributed-training-router/references/notation-and-glossary.md`.

## The HBM equation

```
HBM = model_states + activations + fragmentation + framework_overhead
```

People budget the first two and are surprised by the last two. All four are
real, and the last two are typically 2–8 GB before your model allocates
anything.

| Term | Scales with | Typical size |
|---|---|---|
| model states (weights, grads, optimizer) | `Ψ`, `K`, `1/d` under ZeRO | dominant for small `s` |
| activations | `s·b·h·L` | dominant for large `s` |
| fragmentation | allocation pattern | 0–20% of reserved |
| CUDA context | fixed | ~300–600 MB |
| NCCL buffers | world size, buffer count | ~100 MB – 1 GB+ |
| cuBLAS / cuDNN workspace | kernel selection | ~100 MB – 1 GB |
| PyTorch caching allocator overhead | pattern | included in "reserved" |

## Model states: `2Ψ + 2Ψ + KΨ`

Derived in
`parallelism-strategies/references/data-parallel-and-zero.md`. For
mixed-precision Adam with fp32 master weights and moments, `K = 12` and:

```
model_states = 16Ψ bytes,  or  16Ψ/d under ZeRO-3
```

| Model | `16Ψ` | fits on 80 GB? |
|---|---|---|
| 1.3B | 21 GB | yes |
| 7B | 112 GB | **no** |
| 13B | 208 GB | **no** |
| 70B | 1.12 TB | no |

**"13B bf16 + Adam on an 80GB A100?"** Model states alone are 208 GB. Not on
one GPU — and the answer does not change by lowering the batch size, because
model states do not depend on `b`. Options: shard them (ZeRO-2 needs
`d ≥ 3` for states alone; ZeRO-3 needs `d ≥ 3` with room for activations),
offload the optimizer to CPU, or use an 8-bit optimizer plus bf16 moments.
`references/optimizer-and-precision-memory.md` prices each.

## Activations: `s·b·h·(34 + 5as/h)` per layer

Derived in `references/memory-taxonomy-and-math.md` from arXiv:2205.05198.
Per transformer layer, in bytes, with no parallelism:

```
per layer = s·b·h·(34 + 5as/h)
```

`34` is the attention block (11), MLP (19), and layernorms (4). `5as/h` is the
attention score path — the `O(s²)` term, and the one that explodes at long
context.

With parallelism (same source):

```
TP only:         s·b·h·(10 + 24/t + 5as/(h·t))
TP + SP:         (s·b·h/t)·(34 + 5as/h)
TP + SP + selective recompute:   34·s·b·h/t   per layer
```

Selective recomputation removes the entire `5as/h` term for ~2% overhead — the
best price/performance point on the whole ladder, and the reason it is above
full recompute in the triage order below.

## The OOM triage ladder, priced

Try in this order. Each row states what it costs, in one unit: **added step
time**, or **added communication**.

| # | Action | Frees | Costs |
|---|---|---|---|
| 1 | Lower micro-batch `b`, raise `m` to keep `B` | activations, linearly in `b` | ~0 — often *improves* the comm ratio |
| 2 | Fused attention kernel (if not already) | the `5as/h` term's HBM footprint | negative — it is faster |
| 3 | **Selective** activation recomputation | the `5as/h` term | **~2–5%** step time |
| 4 | ZeRO-1 → ZeRO-2 | `KΨ` then `2Ψ`, ÷`d` | **~0%** — same `2Ψ` traffic as DDP |
| 5 | Sequence parallelism (with TP) | the replicated `10` term | **~0%** — same volume as TP's all-reduces |
| 6 | Full activation recomputation | all activations except layer inputs | **~25–33%** step time |
| 7 | ZeRO-3 / FSDP | `(2Ψ+2Ψ+KΨ)/d` | **+50% comm** (`3Ψ` vs `2Ψ`) |
| 8 | Tensor parallelism | weights and in-block activations, ÷`t` | critical-path all-reduces; requires NVLink |
| 9 | Pipeline parallelism | layers, ÷`p` | bubble `(p−1)/(m+p−1)` |
| 10 | 8-bit optimizer / bf16 moments | `KΨ` from 12 → ~4–6 | small quality risk |
| 11 | CPU offload of optimizer state | `KΨ` entirely | PCIe-bound; see the inequality below |
| 12 | NVMe offload | everything, in principle | very slow; a capability tool only |

Rows 1–5 are nearly free and should all be exhausted before row 6. The single
most common mistake is jumping to full recomputation (row 6, ~30%) when
selective (row 3, ~3%) would have sufficed.

Rows 7–9 are parallelism decisions — the *choice* belongs to
`distributed-train:parallelism-strategies`; they appear here because they are
also capacity tools and belong in the priced ordering.

## *Where* OOM fires is the diagnostic

The step position of the failure identifies the term:

| OOM happens | Term | First move |
|---|---|---|
| Before step 1, during model init | model states | shard (ZeRO), or the model genuinely does not fit |
| In the forward pass | activations | lower `b`, recompute, TP+SP, CP |
| At the backward pass start | peak activations + gradient buffers | recompute; check gradient accumulation buffers |
| At `optimizer.step()` | optimizer state + fp32 master | ZeRO-1/2, 8-bit optimizer, CPU offload |
| During evaluation | eval batch larger than train, or no `no_grad()` | the usual cause is a missing `torch.no_grad()` |
| After N successful steps | fragmentation or a leak | `references/allocator-fragmentation-and-oom-playbook.md` |
| Only at a long sequence | the `5as/h` term | fused attention, then CP |

"OOM after N steps with a constant workload" is almost never a genuine
shortage — it is fragmentation or retained references.

## Fragmentation

The signature is in the error message itself:

```
reserved memory is much greater than allocated memory
```

The caching allocator holds segments it cannot reuse because no free block is
large enough for the requested size, even though the total free bytes suffice.
Causes: varying tensor shapes (variable sequence length), a mix of very large
and very small allocations, and long-lived allocations interleaved with
short-lived ones.

`expandable_segments:True` (a `PYTORCH_CUDA_ALLOC_CONF` option) lets the
allocator grow a segment rather than allocate a new one, which addresses most
of this class. Full playbook, including how to read a memory snapshot:
`references/allocator-fragmentation-and-oom-playbook.md`.

## Offload is a capability tool, not a performance tool

State this plainly whenever offload comes up. The bandwidth inequality:

```
PCIe Gen5 x16   ≈    64 GB/s
HBM (H100)      ≈  3350 GB/s          → ~52× slower
HBM (A100-80)   ≈  2039 GB/s          → ~32× slower
NVMe            ≈  3–7 GB/s           → ~500–1000× slower
```

Anything moved to CPU or NVMe is accessed at a small fraction of HBM speed.
Offload lets you train a model that would not otherwise run. It does not make
training faster, and a plan that assumes it will is wrong.

**Why the optimizer is the right thing to offload.** The optimizer step has
arithmetic intensity ~0.2 FLOP/byte and runs **once per step**, whereas
weights and activations are touched every layer, every micro-batch. Moving
low-intensity, low-frequency work to a slow link is exactly the right trade;
moving the weights is not. ZeRO-Offload (arXiv:2101.06840) is built on this
argument. Details: `references/offload-to-cpu-and-nvme.md`.

## Rules for answering here

- Budget all four terms of the HBM equation, not just weights. State the
  overhead you assumed.
- Price every fix in step time or communication, using the ladder's units.
- Ask (or state) *where* the OOM fires before recommending anything.
- Show the arithmetic with `Ψ`, `K`, `s`, `b`, `h`, `a`, `L`, and the dtype
  stated. A memory estimate without them is unverifiable.
- Say "offload is a capability tool" whenever recommending it.
- Do not emit `forge` commands, and do not attempt to read a job's logs.

## Reference files

| File | Contents |
|---|---|
| `references/memory-taxonomy-and-math.md` | the full HBM equation, the activation formula **derived**, overhead terms, worked budgets for 7B/13B/70B |
| `references/activation-recomputation.md` | the compute/memory trade, full vs selective, the `8N/6N` HFU relation, what to recompute, PP interaction |
| `references/offload-to-cpu-and-nvme.md` | the bandwidth inequality, ZeRO-Offload's argument, ZeRO-Infinity, pinned memory, when offload is and is not right |
| `references/optimizer-and-precision-memory.md` | `K` for every optimizer, 8-bit and factored optimizers, bf16 moments, the fp32 master-weight requirement |
| `references/allocator-fragmentation-and-oom-playbook.md` | the caching allocator, fragmentation signature, `expandable_segments`, memory snapshots, the full OOM playbook |
