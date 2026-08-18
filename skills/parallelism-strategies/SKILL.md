---
name: parallelism-strategies
description: |
  How a model and its state are split across accelerators, and what each split
  costs. Activate for: DDP vs FSDP vs DeepSpeed ZeRO, which ZeRO stage to use,
  tensor parallelism / Megatron column-row splitting, pipeline parallelism and
  the bubble, GPipe / 1F1B / interleaved / zero-bubble schedules, sequence vs
  context parallelism, ring attention, expert parallelism and MoE routing,
  choosing TP/PP/DP/CP degrees, 3D or 4D parallelism mesh design, "TP=8 or
  TP=4 PP=2", "왜 pipeline parallelism에 bubble이 생겨", "DDP랑 FSDP 차이",
  "70B 모델 어떻게 쪼개", parallelism ordering across nodes, and comm-volume
  comparisons between sharding strategies.
  Do NOT activate for: the collective algorithms and interconnect underneath
  (ring vs tree all-reduce, NCCL mechanics, bucketing, hangs, stragglers — use
  distributed-train:communication-backends); how much HBM a config needs, OOM
  triage, activation checkpointing or offload (use
  distributed-train:memory-offloading); MFU, throughput or loss curves (use
  distributed-train:training-metrics); single-GPU kernels, roofline or
  precision (use distributed-train:gpu-architecture).
  Also do NOT activate for running or inspecting real cluster jobs — submitting
  a training job, GPU quota, job status, reading a failed job's logs, multi-node
  launch, images or disks — that is mlops:forge-train.
---

# Parallelism Strategies

Which dimension of the problem gets split, across which accelerators, and
what each split costs per step. This skill decides *shape*; the wire cost of
that shape is `distributed-train:communication-backends`, and whether it fits
in HBM is `distributed-train:memory-offloading`.

> Platform recipe: launching any of this on a real cluster — node counts,
> launcher flags, images, multi-node submission — is skill `mlops:forge-train`.
> This skill stops at the configuration, not the command.

Symbols (`Ψ`, `n`, `L`, `h`, `a`, `s`, `b`, `m`, `B`, `d`, `t`, `p`, `c`,
`e`, `v`) and the two invariants `n = d·t·p·c` and `B = b·m·d` are defined in
skill `distributed-train:distributed-training-router`, file
`distributed-training-router/references/notation-and-glossary.md`. Do not redefine them.

## The four things you can split

Every parallelism scheme splits exactly one of these, and combinations split
several at once:

| Split | Axis | Scheme | Per-rank state | Per-rank activations |
|---|---|---|---|---|
| the **batch** | `B` | DP / ZeRO / FSDP | full (DP) or `1/d` (ZeRO) | `1/d` |
| a **layer's weights** | `h`, ffn | TP (Megatron) | `1/t` | `1/t` inside the block |
| the **layer stack** | `L` | PP | `1/p` | `1/p` of layers, but ×`p` in flight |
| the **sequence** | `s` | SP / CP / ring attention | full | `1/c` |
| the **experts** | MoE | EP | `1/e` of expert weights | routed |

Reading them as "what gets divided by what" is the whole design space. What
distinguishes the schemes in practice is not the division — it is the
communication each one forces, per layer, per step.

## Communication volume — the table that decides everything

Bytes moved per rank, per optimizer step, expressed in elements. Multiply by
bytes-per-element for the dtype actually on the wire. Derivations in
`references/data-parallel-and-zero.md` and the per-scheme files.

| Scheme | Collective(s) | Volume per step | Note |
|---|---|---|---|
| DDP | 1 all-reduce on gradients | `2Ψ` | `2(n−1)/n · Ψ ≈ 2Ψ`; fully overlappable with backward |
| ZeRO-1 | reduce-scatter grads + all-gather params | `2Ψ` | **same as DDP** — RS∘AG = AR |
| ZeRO-2 | reduce-scatter grads + all-gather params | `2Ψ` | still `2Ψ`; gradients also sharded |
| ZeRO-3 / FSDP | AG params (fwd) + AG params (bwd) + RS grads | `3Ψ` | the **1.5× rule** |
| TP (Megatron) | 4 all-reduces per layer per step | `4 · b·s·h` per layer | 2 forward, 2 backward, on activations |
| PP | point-to-point at each boundary | `2·b·s·h` per boundary | tiny volume, but it *serializes* |
| CP (ring attention) | `c−1` ring steps of K/V per attention | `2·b·s·h/c` per step × `(c−1)` | overlappable with attention compute |
| EP (MoE) | 2 all-to-all per MoE layer | routed tokens × `h` | volume depends on routing balance |

Three consequences fall straight out of this table, and they are the reason
the ordering heuristic below is *derived* rather than asserted:

1. **ZeRO-1 and ZeRO-2 are free.** They shard optimizer state (and gradients)
   with the *same* `2Ψ` of traffic as DDP, because an all-reduce is exactly a
   reduce-scatter followed by an all-gather. If you are running DDP and
   memory is tight, ZeRO-2 is a strict improvement.
2. **ZeRO-3 costs 50% more wire traffic** (`3Ψ` vs `2Ψ`), because parameters
   must be gathered twice — once for forward, once for backward. That is the
   price of the largest capacity win.
3. **TP volume scales with activations, not parameters.** `4·b·s·h` per layer
   per step is enormous at long sequence and is paid `L` times. This is why
   TP must stay inside the NVLink domain.

## The ordering heuristic, derived

Rank the axes by bytes on the wire per unit of compute, then assign the
noisiest axis to the fastest link:

```
TP:  4·b·s·h per layer  × L layers  → ~4·L·b·s·h  per step, on activations
CP:  ring K/V, (c−1) × 2·b·s·h/c    → ~2·b·s·h    per attention, overlappable
PP:  2·b·s·h per boundary × (p−1)   → ~2·p·b·s·h  per step, but p2p and rare
DP:  2Ψ (or 3Ψ) once per step       → independent of s, overlappable
```

For a 70B-class model at `s=8192`, TP's per-step volume exceeds DP's by an
order of magnitude, and it cannot be hidden — each of the four all-reduces
sits on the critical path between two matmuls of the same layer.

Therefore:

```
TP  →  innermost, inside one node, over NVLink.   t ≤ GPUs per node.
CP  →  next, also intra-node when it fits.
PP  →  across nodes; p2p volume is small and the links are slow.
DP  →  outermost; its all-reduce overlaps with backward compute.
```

`references/composing-nd-parallelism.md` works three fully-numeric mesh
configurations end to end, including the arithmetic that rejects the
alternatives.

## Pipeline bubbles

The idle fraction of a pipeline stage, for `m` micro-batches over `p` stages:

| Schedule | Bubble ratio | Activation memory |
|---|---|---|
| GPipe (all-forward-all-backward) | `(p−1)/(m+p−1)` | `p` micro-batches in flight |
| 1F1B (PipeDream-Flush) | `(p−1)/(m+p−1)` | `p` in flight, but bounded and steady |
| Interleaved 1F1B, `v` virtual stages | `(1/v)·(p−1)/m` | `v`× more p2p messages |
| Zero Bubble (arXiv:2401.10241) | → ~0 | splits backward into input-grad and weight-grad |

The bubble is a **serialization** cost, not a bandwidth cost — the links are
idle during it. The lever is `m`: at `p=8`, going from `m=8` to `m=64` cuts
the GPipe bubble from 47% to 10%. But `m` is bounded by `B = b·m·d`, so
increasing `m` at fixed `B` means shrinking `d` — the bubble and the
data-parallel width trade directly against each other.

Full schedules, the `f`/`f̄` conjugate pair for TP, and the memory
consequences are in `references/pipeline-parallel.md`.

## Sequence parallelism ≠ context parallelism

These are routinely conflated. They solve different problems:

| | Sequence parallelism (Megatron) | Context parallelism (ring attention) |
|---|---|---|
| What is sharded | layernorm and dropout regions only | the whole sequence, attention included |
| Attention | unchanged, full `s` per rank | K/V circulated between ranks |
| Extra collectives | all-gather / reduce-scatter replacing TP's all-reduces | `c−1` ring steps of K/V |
| Motivation | remove the replicated `10·s·b·h` term TP leaves behind | fit `s` that no single rank can hold |
| Composes with | TP, at the same degree | TP, PP, DP as its own mesh axis |

`references/sequence-context-parallel.md` separates them properly and covers
ring attention, the causal-mask load imbalance, and Striped Attention
(arXiv:2311.09431) as the fix for it.

## Choosing a configuration

The order that converges fastest, because each answer constrains the next:

1. **Does it fit under ZeRO-2?** ZeRO-2 is free in bandwidth. If the model
   states fit at `1/d`, stop — you do not need model parallelism.
   (`memory-offloading` owns the fit calculation.)
2. **Still short?** Add ZeRO-3/FSDP and accept the `3Ψ` traffic, *or* add TP
   if you have NVLink and the activations are the problem rather than the
   states. TP shrinks activations too; ZeRO-3 does not.
3. **Model still too big for one node?** Add PP. Choose `p` to be the fewest
   stages that make it fit, because bubble grows with `p`.
4. **Sequence too long?** Add CP. It is the only axis that shrinks the
   `O(s²)` attention working set across ranks.
5. **MoE?** EP replaces most of what TP would do for the expert weights, but
   introduces all-to-all — see `references/expert-parallel-moe.md`.
6. **Fill the rest with DP.** `d = n/(t·p·c)`.

Then check both invariants (`n = d·t·p·c`, `B = b·m·d`) and check that `m` is
large enough that the bubble is tolerable. If it is not, the config is wrong
even if it fits.

## Rules for answering here

- Give the mesh as a full tuple `(d, t, p, c)` with `n = d·t·p·c` shown, and
  state `B = b·m·d` explicitly. A recommendation missing either is unchecked.
- Price every recommendation in the comm-volume table's units. "Use ZeRO-3"
  without "+50% wire traffic vs ZeRO-2" is half an answer.
- Say which budget the change addresses (capacity / bandwidth / serialization)
  — the router's frame.
- Distinguish SP from CP explicitly whenever either appears.
- Do not emit `forge` commands, launcher recipes, or cluster-specific env-var
  values.

## Reference files

| File | Contents |
|---|---|
| `references/data-parallel-and-zero.md` | DDP mechanics, the `2Ψ+2Ψ+KΨ` derivation, ZeRO stages 1/2/3, FSDP vs DeepSpeed, the RS∘AG identity, the 1.5× rule |
| `references/tensor-parallel.md` | Megatron column→row split, the `f`/`f̄` conjugate pair, why 4 all-reduces, attention head splitting, `t ≤` node size |
| `references/pipeline-parallel.md` | GPipe / 1F1B / interleaved / zero-bubble, bubble derivations, stage balancing, p2p volume, PP + recompute interaction |
| `references/sequence-context-parallel.md` | Megatron SP vs ring-attention CP, the ring schedule, causal load imbalance, Striped Attention |
| `references/expert-parallel-moe.md` | top-k routing, `N_total` vs `N_active`, all-to-all cost, capacity factor and dropping, expert-choice routing, Tutel |
| `references/composing-nd-parallelism.md` | the ordering derivation, three fully-numeric mesh configurations, degrees-of-freedom counting, published configs |
