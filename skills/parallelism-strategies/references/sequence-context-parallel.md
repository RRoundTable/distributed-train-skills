# Sequence Parallelism vs Context Parallelism

Two different things with confusingly similar names. Symbols: router →
`notation-and-glossary.md`.

## The distinction, stated once

| | **Sequence parallelism (SP)** — Megatron's sense | **Context parallelism (CP)** — ring attention's sense |
|---|---|---|
| Motivating problem | TP leaves layernorm/dropout activations replicated | the sequence itself is too long for one rank |
| What is sharded along `s` | *only* the non-tensor-parallel regions (LN, dropout, residual) | **everything**, attention included |
| Attention computation | untouched; each rank still sees full `s` | K/V blocks circulate; each rank sees `s/c` queries |
| Degree | tied to `t` — it is a refinement of TP | independent mesh axis `c` |
| Extra collectives | all-gather + reduce-scatter *replacing* TP's all-reduces (same volume) | `c−1` ring steps exchanging K/V |
| Counts in the mesh invariant | no (it is TP) | **yes**: `n = d·t·p·c` |

If someone says "sequence parallelism" and means fitting a 1M-token context,
they mean CP. If they say it in a Megatron config next to `tensor_model_parallel_size`,
they mean SP.

The CP column above describes the **ring** family. There is a second family —
DeepSpeed-Ulysses — that shards `s` the same way but reaches full attention by
transposing to a head-sharded layout instead of circulating K/V. It is a
different set of trade-offs, not a different goal; both are covered below, with
a comparison table. Confusingly, DeepSpeed calls it "sequence parallelism"
too, which is a third use of the phrase.

## Sequence parallelism (Megatron)

TP shards the attention and MLP blocks but leaves layernorm, dropout, and the
residual path replicated across all `t` ranks. From the activation formula
(constants from arXiv:2205.05198, derived in
`memory-offloading/references/memory-taxonomy-and-math.md`):

```
TP only:     s·b·h·(10 + 24/t + 5as/(h·t))
TP + SP:     (s·b·h/t)·(34 + 5as/h)
```

The `10` is exactly the replicated region. At `t = 8`, TP-only leaves
`10 + 3 = 13` units where TP+SP leaves `34/8 = 4.25` — a 3× difference in the
non-attention term.

SP shards those regions along `s`: rank `j` owns tokens
`[j·s/t, (j+1)·s/t)` for the layernorm and dropout. Since layernorm is
per-token, no communication is needed *within* the operation. What changes is
the boundary:

```
TP alone:      ... → [TP region] → all-reduce → [replicated LN] → ...
TP + SP:       ... → [TP region] → reduce-scatter → [SP region] → all-gather → [TP region] → ...
```

`AllReduce = ReduceScatter ∘ AllGather`, so **the total volume is unchanged**
(`communication-backends/references/collective-algorithms.md`). SP buys a
large activation-memory reduction for zero extra bandwidth. There is no
reason to run TP without SP on a modern stack.

## Context parallelism / ring attention

When `s` is large enough that the attention working set does not fit on one
rank, shard the sequence itself across `c` ranks. Rank `j` holds queries
`Q_j` (its `s/c` tokens) permanently, and the K/V blocks rotate around a
ring:

```
for step in 0 .. c-1:
    partial = attention(Q_j, K_recv, V_recv)     # local compute
    accumulate partial into the running output   # online softmax
    send (K_recv, V_recv) to rank j+1
    recv next (K, V) from rank j-1
```

Two properties make this work:

1. **Online softmax.** You cannot normalize until you have seen every key, but
   you can maintain a running max `M` and running sum `Z` and rescale the
   accumulator as new blocks arrive. This is the same recurrence FlashAttention
   uses; it is derived once in
   `gpu-architecture/references/kernel-fusion-and-flash-attention.md`.
2. **Overlap.** The send of block `i` overlaps the compute on block `i`, so
   with enough work per block the communication is nearly free. Per ring step
   each rank sends `2·b·(s/c)·h` elements (K and V), and there are `c−1`
   steps — total `≈ 2·b·s·h·(c−1)/c` per attention, per step.

Memory per rank for the attention working set drops from `O(s²)` terms to
`O((s/c)·s)` per block, and activations generally scale as `1/c`.

## The causal-mask imbalance — and Striped Attention

With a causal mask, query token `i` attends only to keys `≤ i`. If the
sequence is split contiguously — rank 0 gets tokens `[0, s/c)`, rank `c−1`
gets `[s−s/c, s)` — then rank 0's queries attend to almost nothing and rank
`c−1`'s attend to almost everything. Work per rank is linear in position, so
the last rank does `~c×` the work of the first, and the ring runs at the
speed of the slowest.

Two standard fixes:

- **Load-balanced (zigzag) sharding.** Split the sequence into `2c` chunks
  and give rank `j` chunks `j` and `2c−1−j`. Each rank now holds one early
  and one late chunk, equalizing the causal work. This is what most
  implementations do by default.
- **Striped Attention** (arXiv:2311.09431). Assign tokens to ranks by
  *stride* rather than by block — rank `j` gets tokens `j, j+c, j+2c, …`. The
  causal structure then distributes evenly across ranks by construction, and
  the paper reports substantial end-to-end speedups over plain ring attention
  for causal models.

Checking for this imbalance is easy and worth doing: if per-rank step time
increases monotonically with rank index under CP, the sharding is contiguous
and unbalanced.

## DeepSpeed-Ulysses — shard `s`, then transpose to heads

Ring attention moves the **data** to where the queries are. Ulysses
(arXiv:2309.14509) moves the **layout** instead, and the reason it can is a
property of attention worth stating on its own:

```
attention needs every token, but it is completely independent across heads.
```

So instead of circulating K/V, re-shard: go from "every rank has 1/c of the
tokens and all heads" to "every rank has all the tokens and 1/c of the heads",
compute attention locally, then go back.

```
input, sharded along s:   [s/c tokens, all heads]
        ↓ QKV projection                       local, no communication
all-to-all #1:            [all s tokens, a/c heads]
        ↓ attention                            local, full sequence, subset of heads
all-to-all #2:            [s/c tokens, all heads]
        ↓ MLP, layernorm                       local, sharded by s again
```

The attention call in the middle sees a complete, ordinary sequence. That is
the design's most useful consequence: **the attention kernel is untouched.**
FlashAttention v2, sparse attention, or anything else drops in unmodified,
because Ulysses never splits an attention computation — it only decides who
holds which heads. Ring attention, by contrast, requires the kernel to
accumulate partial results across blocks with online softmax.

### The communication argument

Per transformer layer, the aggregate message is `3·b·s·h` for Q, K, V plus
`b·s·h` for the output — `4·b·s·h`. The paper's claim rests on all-to-all's
per-link property: each rank sends `1/c` of that to each peer, so

```
Ulysses      per-link volume  =  4·b·s·h / c      →  O(s·h/c)
Megatron SP  per-link volume  =  4·b·s·h          →  O(s·h)
```

a `c`-fold reduction, and — the headline framing — **per-link volume stays
constant when sequence length and device count grow together.** Doubling `s`
while doubling `c` leaves each link moving the same bytes.

Two caveats keep this honest:

- It is a **per-link** comparison. All-to-all is the collective least able to
  exploit topology (`communication-backends/references/collective-algorithms.md`):
  `n²` distinct messages, no ring or tree structure, most exposed of any
  collective to a hierarchical fabric. The `O(·)` hides that, and it is the
  reason Ulysses is happiest when the sequence-parallel group stays inside a
  node.
- There are **two hard synchronization points per attention**, whereas the
  ring's sends overlap its own compute block by block. Ulysses trades
  overlappability for volume.

### The ceiling that decides between them

Each rank must own a whole, non-overlapping subset of heads. Therefore:

```
c  ≤  number of attention heads       (and must divide it)
```

Under grouped-query attention the binding number is the **KV head count**,
which is much smaller — 8 in several current models. This constraint is not
stated in the paper; it falls out of the mechanism, and it is the single most
important practical difference from ring attention, which has no such ceiling
and can scale `c` past the head count.

### Reported results

Up to 256 A100s; 7B and 30B GPT models on 32 and 64 GPUs. **4× longer
sequences** than the compared systems, over **1M tokens** on a 1.2B model,
throughput up to **2.5×** the Megatron-LM baseline, and sustained **175+
TFLOP/s per GPU — "over 54% of hardware peak"**. That last figure is worth a
check: A100's dense bf16 peak is 312 TFLOP/s, and `175/312 = 56%`, so the
paper is quoting against the **dense** peak. It passes the audit in
`training-metrics/references/flop-counting-and-mfu.md`.

ZeRO-3 composes with it by partitioning model states across the *combined*
data-and-sequence-parallel group.

### Ring vs Ulysses

| | Ring attention / CP | Ulysses |
|---|---|---|
| What moves | K/V blocks, `c−1` steps | the layout — 2 all-to-alls per attention |
| Per-link volume per layer | `≈ 2·b·s·h` | `4·b·s·h / c` |
| Degree ceiling | none — `c` may exceed head count | **`c ≤` head count** (KV heads under GQA) |
| Causal load balance | broken by default; needs zigzag or Striped | intrinsic — every rank holds the whole sequence |
| Attention kernel | must accumulate across blocks (online softmax) | **unmodified** — any kernel |
| Overlap | sends hide behind per-block compute | two synchronization points per attention |
| Topology sensitivity | ring, degrades gracefully | all-to-all, most exposed |

They shard different dimensions — heads versus sequence blocks — so the two
are composable in principle, and hybrid implementations exist: use Ulysses up
to the head-count ceiling, then a ring across groups beyond it.

## Composing CP with the other axes

CP is a full mesh axis: `n = d·t·p·c`. Practical guidance:

- **CP with TP.** Both are intra-node-preferring. On an 8-GPU node, `t·c ≤ 8`.
  Llama 3 405B uses the order `[TP, CP, PP, DP]` (arXiv:2407.21783), i.e. TP
  innermost, then CP, then PP, then DP outermost — the same ordering the
  comm-volume argument in `references/composing-nd-parallelism.md` predicts.
- **CP with TP, under Ulysses specifically: they compete for the same
  resource.** Megatron TP splits attention heads across `t` ranks
  (`references/tensor-parallel.md`), and Ulysses splits heads across `c`
  ranks. Both consume head count, so

  ```
  t · c  ≤  a          (KV heads under GQA, which is far fewer)
  ```

  A model with 8 KV heads and `t = 8` has **no heads left** for Ulysses. That
  configuration needs the ring, and it is a common one — which is why the
  ceiling is worth checking before the mesh is chosen rather than after.
- **CP with DP.** `B = b·m·d` still holds; `c` does not multiply the batch.
  Ranks in a CP group process the *same* samples.
- **CP and MFU.** Llama 3 reports MFU falling to 38% at 131k context, because
  CP replaces FSDP's well-overlapped all-reduce with the ring's rolling K/V
  exchange, which has less compute to hide behind at that shape.

## When to reach for which

| Situation | Use |
|---|---|
| Running TP and activations are tight | **SP** — free, always on |
| `s` fits on one rank but activation memory is tight | recomputation or TP+SP before CP |
| `s` is too long for one rank at any `b` | **CP** — then pick the family below |
| Long context *and* a big model | TP+SP inside the node, CP next, PP across nodes |
| Inference with very long context | CP, but the tradeoffs differ — this file is about training |

And within CP:

| Situation | Use |
|---|---|
| `t · c` would exceed the (KV) head count | **ring** — Ulysses has no heads left to shard |
| Needed degree exceeds the head count | **ring** — the only one that scales past it |
| The sequence-parallel group fits inside a node | **Ulysses** — all-to-all is cheap on NVLink, and volume is `c`× lower |
| You want an unmodified attention kernel (custom, sparse, newest FlashAttention) | **Ulysses** — attention stays a black box |
| Causal model and you cannot control the sharding order | **Ulysses** — no causal imbalance to fix |
| Group spans many nodes | **ring** — all-to-all across a fabric is the worst case |

## Sources

- Reducing Activation Recomputation in Large Transformer Models (sequence parallelism) — arXiv:2205.05198
- DeepSpeed-Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models — arXiv:2309.14509
- Striped Attention: Faster Ring Attention for Causal Transformers — arXiv:2311.09431
- The Llama 3 Herd of Models (4D parallelism, CP at 131k context) — arXiv:2407.21783
- FlashAttention (online softmax) — arXiv:2205.14135
