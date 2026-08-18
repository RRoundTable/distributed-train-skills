# Offload to CPU and NVMe

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`.

## The inequality that governs everything here

```
HBM (H100 SXM)     3350 GB/s
HBM (A100 80GB)    2039 GB/s
NVLink (H100)       900 GB/s
PCIe Gen5 x16        ~64 GB/s     ← the CPU offload path
PCIe Gen4 x16        ~32 GB/s
NVMe SSD             3–7 GB/s     ← the NVMe offload path
```

CPU offload runs at **~1/30 to 1/50** of HBM bandwidth. NVMe at **~1/500 to
1/1000**.

**Offload is a capability tool, not a performance tool.** It lets a model run
that otherwise would not. It does not make training faster, and any plan that
assumes it will is wrong. Say this explicitly whenever recommending it.

## Why the optimizer is the right thing to offload

Two properties decide whether a tensor is a good offload candidate: how often
it is touched, and how much compute is done per byte moved.

| State | Touched | Arithmetic intensity | Offload? |
|---|---|---|---|
| Optimizer state (`KΨ`) | once per **step** | ~0.2 FLOP/byte | **yes** |
| fp32 master weights | once per step | ~0.2 | **yes**, with the optimizer |
| Gradients | once per step (after reduce) | ~0.2 | maybe |
| Weights (bf16) | every layer, every micro-batch | high (they feed GEMMs) | **no** |
| Activations | every layer, every micro-batch | n/a | **no** — and they are the largest term at long `s` |

The optimizer step reads and writes ~14 bytes per parameter to perform ~10
FLOPs — it is the single most memory-bound, least compute-dense part of the
step, and it happens **once**, not `L·m` times. That combination is exactly
what you want to move to a slow link: minimal work per byte, minimal
frequency.

ZeRO-Offload (arXiv:2101.06840) is built on this argument. It keeps the
forward/backward entirely on GPU and moves the optimizer step — state and
computation — to the CPU, exploiting the fact that CPUs are perfectly capable
of a memory-bound elementwise update.

## The cost, quantified

Per step, offloading the optimizer moves gradients down and updated parameters
up:

```
down:  gradients          2Ψ bytes (bf16)   or 4Ψ (fp32)
up:    updated params     2Ψ bytes (bf16)
total ≈ 4Ψ bytes over PCIe per step, per GPU (before sharding)
```

For a 7B model: `4 · 7e9 = 28 GB` over PCIe Gen5 at ~64 GB/s ≈ **0.44 s per
step**, per GPU — and PCIe is often shared between GPUs on a node, so the
effective per-GPU share can be much lower and the time correspondingly larger.
Against a 2-second step that is a 20%+ penalty; against a 0.5-second step it
is catastrophic.

Two mitigations that matter:

- **Overlap.** Transfer gradients for layer `i` while layer `i−1` is still in
  backward. ZeRO-Offload does this; it converts much of the transfer into
  hidden time.
- **Shard first.** Under ZeRO, each rank only owns `1/d` of the optimizer
  state, so each rank offloads `1/d` of the traffic. Offload composes with
  sharding rather than replacing it, and should be applied *after* it.

## Pinned memory is not optional

Transfers to/from **pageable** host memory require the driver to stage through
an internal pinned buffer, roughly halving throughput and preventing
asynchronous overlap. Offload paths must use pinned (page-locked) host memory.

The cost: pinned pages cannot be swapped out, so over-allocating them degrades
the whole host. Frameworks pre-allocate a bounded pinned buffer pool for this
reason, which is also why offload configurations have a buffer-count knob.

## NVMe offload and ZeRO-Infinity

ZeRO-Infinity (arXiv:2104.07857) extends the hierarchy to NVMe, making it
possible to train models whose state exceeds aggregate CPU DRAM. The design
adds bandwidth-centric partitioning (spreading a tensor's traffic across all
the ranks' NVMe devices rather than one) and overlapping prefetch.

At 3–7 GB/s per device, this is unambiguously a capability tool. It answers
"can this model be trained at all on this hardware" with a yes, at a
throughput that makes it suitable for research access to a scale otherwise
unavailable — not for a production training run where GPU-hours are the
constraint.

## Activation offload

Activations can also be moved to host memory after the forward pass and
brought back during backward. It is much less attractive than optimizer
offload:

- Activations are touched during **every** layer's backward, so the traffic is
  `O(L)` per micro-batch rather than `O(1)` per step.
- Activation memory is often larger than optimizer state at long sequence.
- Recomputation achieves the same memory saving for ~2–33% compute rather than
  a PCIe round trip per layer.

Prefer recomputation (`references/activation-recomputation.md`). Activation
offload is worth considering only when compute is genuinely idle — which, if
true, is itself a finding worth investigating.

## Decision procedure

```
Does it fit with ZeRO-3 + selective recompute + the parallelism you can afford?
    yes → do that. No offload.
    no  ↓
Is the shortfall in optimizer state?
    yes → CPU offload of the optimizer. Expect a step-time penalty; measure it.
    no  ↓
Is the shortfall in activations?
    yes → recomputation, TP+SP, or CP. NOT activation offload.
    no  ↓
Is the shortfall in weights themselves, even sharded?
    yes → more GPUs, or ZeRO-Infinity/NVMe if the goal is feasibility not speed.
```

## Measuring whether offload is working

| Signal | Reading |
|---|---|
| Step time up by roughly the predicted PCIe time | working as designed |
| Step time up much more than predicted | transfers not overlapped, or pageable (unpinned) memory |
| `Memcpy DtoH`/`HtoD` visible as long serial blocks in a trace | no overlap — check the framework's overlap settings |
| Host CPU saturated | the CPU-side optimizer step is the bottleneck, not PCIe |
| Host memory pressure / swapping | pinned buffer pool too large, or state larger than DRAM |

The fourth row surprises people: with the optimizer computation moved to the
CPU, a weak host CPU becomes the bottleneck rather than PCIe. Check both
before tuning either.

> Platform recipe: how much host RAM or NVMe a given node type has, and how to
> request it, is skill `mlops:forge-train`.

## Sources

- ZeRO-Offload: Democratizing Billion-Scale Model Training — arXiv:2101.06840
- ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning — arXiv:2104.07857
