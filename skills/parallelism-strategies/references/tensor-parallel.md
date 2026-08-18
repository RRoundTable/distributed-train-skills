# Tensor Parallelism

Splitting the weights of a *single* layer across `t` ranks so that one matmul
becomes `t` smaller matmuls. Symbols: router → `notation-and-glossary.md`.

## The core trick: column then row

A transformer MLP is `Y = GeLU(XA) · B`. Naively, splitting `A` by rows would
force a reduction *before* the nonlinearity, since GeLU is not linear. Megatron
(arXiv:1909.08053) avoids this by splitting the two matrices in complementary
directions:

```
A split by COLUMNS:   A = [A₁ | A₂]      →  XA = [XA₁ | XA₂]
   GeLU applies elementwise, so  GeLU([XA₁ | XA₂]) = [GeLU(XA₁) | GeLU(XA₂)]
   → no communication needed before the nonlinearity

B split by ROWS:      B = [B₁ ; B₂]
   → Y = GeLU(XA₁)·B₁ + GeLU(XA₂)·B₂     (a partial sum on each rank)
   → one all-reduce produces Y
```

Column-split first, row-split second: **one** all-reduce per MLP block in the
forward pass, placed after the second matmul. The nonlinearity is crossed
without communication.

Attention uses the same idea with a natural split point. `Q`, `K`, `V`
projections are column-parallel *by head* — rank `j` owns heads
`[j·a/t, (j+1)·a/t)` and computes their attention completely locally, since
attention never mixes heads. The output projection is row-parallel, so again
**one** all-reduce per attention block.

## The `f` / `f̄` conjugate pair

Megatron expresses this as a pair of operators inserted around each block:

| Operator | Forward | Backward |
|---|---|---|
| `f` (entering the block) | identity | all-reduce |
| `f̄` (leaving the block) | all-reduce | identity |

They are conjugates: whatever one does in forward, the other does in
backward. Per transformer layer that gives:

```
forward:   1 all-reduce (attention f̄) + 1 all-reduce (MLP f̄)   = 2
backward:  1 all-reduce (attention f)  + 1 all-reduce (MLP f)   = 2
                                                       total  = 4 per layer
```

Each all-reduce moves an **activation** tensor of shape `b × s × h`, so the
per-layer, per-step volume is:

```
V_TP  ≈  4 · b · s · h  elements per layer
       (busbw-corrected: 4 · 2(t−1)/t · b·s·h)
```

Across `L` layers: `≈ 4·L·b·s·h`. For `L=80`, `b=1`, `s=8192`, `h=8192` in
bf16 that is `4·80·1·8192·8192·2 bytes ≈ 43 GB` moved per step, per rank.
Over NVLink at ~450 GB/s effective that is ~95 ms. Over 400 Gb/s InfiniBand
(50 GB/s) it would be ~860 ms — which is why the rule below is not a
preference.

## Why `t` must not cross a node boundary

Compare TP's per-layer traffic with DP's per-step traffic:

- TP: `4·b·s·h` per layer, **on the critical path** — the next matmul of the
  same layer cannot start until the all-reduce completes.
- DP: `2Ψ` once per step, **overlappable** with backward compute.

TP traffic is both larger (at realistic `s`) and unhideable. Hence:

```
t ≤ (number of GPUs in one NVLink domain)
```

On an 8-GPU NVLink node, `t ∈ {1,2,4,8}`. Setting `t = 16` across two nodes
puts a critical-path all-reduce on the slow fabric `4L` times per step and
typically costs more than it saves. The exception is a fabric where inter-node
bandwidth approaches NVLink (some NVLink-switched multi-node systems), where
the rule relaxes but the *ordering* argument does not.

## What TP buys

Per-rank, TP divides:

| Quantity | Factor |
|---|---|
| layer weights (attention + MLP) | `1/t` |
| gradients and optimizer state for those weights | `1/t` |
| activations *inside* the parallel regions | `1/t` |
| layernorm/dropout activations | **1** (replicated) |

That last row is the gap Megatron's sequence parallelism closes. Using the
activation-memory formula derived in
`memory-offloading/references/memory-taxonomy-and-math.md`, per layer:

```
no parallelism:      s·b·h·(34 + 5as/h)
TP only:             s·b·h·(10 + 24/t + 5as/(h·t))
TP + SP:             (s·b·h/t)·(34 + 5as/h)
```

The `10` is the replicated layernorm and dropout term: at `t=8` it is over
40% of what remains. Sequence parallelism shards it along `s` and converts
TP's all-reduces into an all-gather/reduce-scatter pair of *the same total
volume* — so SP is essentially free memory. See
`references/sequence-context-parallel.md`. (Constants from arXiv:2205.05198.)

## Practical constraints on `t`

| Constraint | Requirement |
|---|---|
| head splitting | `a` divisible by `t` (and, for GQA, `a_kv` divisible by `t` — often the binding one) |
| FFN split | intermediate size divisible by `t` |
| vocabulary parallelism | `V` divisible by `t`; pad `V` if not |
| tile efficiency | per-rank matmul dims should stay multiples of 128 — `t` too large makes each shard too small to fill tensor cores (see `gpu-architecture`) |
| node topology | `t` ≤ NVLink domain size |

Grouped-query attention is the common trap: a model with `a = 64` query heads
but `a_kv = 8` key/value heads cannot use `t = 16` head-parallel without
replicating K/V.

## Vocabulary parallelism

The embedding and output projection are `V × h`, and `V` is large enough
(32k–128k) to be worth splitting. Megatron column-splits the output layer by
vocabulary and fuses the cross-entropy so that only the *scalar* loss and the
per-token max/sum need reducing, rather than the full `b·s·V` logits tensor.
Without that fusion the logits all-gather alone can exceed every other
collective in the step: at `b·s = 8192` and `V = 128256` in bf16 that is
2.1 GB.

## Diminishing returns

TP's compute per rank falls as `1/t` but its communication per rank falls
only as `2(t−1)/t → 2` — i.e. it stops falling. The compute-to-communication
ratio therefore degrades roughly linearly in `t`. Empirically, efficiency is
good at `t ∈ {2,4,8}` and poor beyond, which is why every published large
config uses `t ≤ 8` and reaches for PP or ZeRO-3 instead of `t = 16`.

## When to use TP rather than ZeRO-3

Both shard parameters. They differ in what else they do:

| | TP | ZeRO-3 / FSDP |
|---|---|---|
| shards activations | **yes**, inside parallel regions | no |
| traffic scales with | `b·s·h` per layer | `Ψ` per step |
| traffic overlappable | poorly (critical path) | well (prefetch) |
| needs fast interconnect | yes, NVLink | tolerates fabric |
| changes the model code | yes (parallel layers) | no |

Rule of thumb: if the **activations** are what will not fit, TP (with SP)
helps and ZeRO-3 does not. If the **model states** are what will not fit and
you have only fabric between ranks, ZeRO-3 is the better tool. Large runs use
both, with TP inside the node and ZeRO/DP outside.

## Sources

- Megatron-LM — arXiv:1909.08053
- Efficient Large-Scale LM Training on GPU Clusters (PTD-P) — arXiv:2104.04473
- Reducing Activation Recomputation in Large Transformer Models — arXiv:2205.05198
