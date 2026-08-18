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

## Composing CP with the other axes

CP is a full mesh axis: `n = d·t·p·c`. Practical guidance:

- **CP with TP.** Both are intra-node-preferring. On an 8-GPU node, `t·c ≤ 8`.
  Llama 3 405B uses the order `[TP, CP, PP, DP]` (arXiv:2407.21783), i.e. TP
  innermost, then CP, then PP, then DP outermost — the same ordering the
  comm-volume argument in `references/composing-nd-parallelism.md` predicts.
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
| `s` is too long for one rank at any `b` | **CP** |
| Long context *and* a big model | TP+SP inside the node, CP next, PP across nodes |
| Inference with very long context | CP, but the tradeoffs differ — this file is about training |

## Sources

- Reducing Activation Recomputation in Large Transformer Models (sequence parallelism) — arXiv:2205.05198
- Striped Attention: Faster Ring Attention for Causal Transformers — arXiv:2311.09431
- The Llama 3 Herd of Models (4D parallelism, CP at 131k context) — arXiv:2407.21783
- FlashAttention (online softmax) — arXiv:2205.14135
