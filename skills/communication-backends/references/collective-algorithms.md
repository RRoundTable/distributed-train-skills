# Collective Algorithms

Symbols and the `busbw`/`algbw` convention: router →
`distributed-training-router/references/notation-and-glossary.md`. Cost model: `T(S) = α + β·S`.

## Ring all-reduce, derived

`n` ranks each hold a vector of `S` bytes. Every rank must end with the
element-wise sum. Arrange the ranks in a ring; split each vector into `n`
chunks of `S/n` bytes.

### Phase 1 — reduce-scatter (`n−1` steps)

At step `k` (`k = 0 … n−2`), rank `i` sends chunk `(i − k) mod n` to rank
`i+1` and receives chunk `(i − k − 1) mod n` from rank `i−1`, adding it to
its local copy.

After `n−1` steps, rank `i` holds the **fully reduced** chunk `(i+1) mod n`,
and no rank holds any other chunk completely.

```
steps: n−1     bytes sent per rank: (n−1)·S/n
cost:  (n−1)·α + (n−1)/n · S · β
```

### Phase 2 — all-gather (`n−1` steps)

The same rotation, but now copying rather than accumulating: each rank
forwards the fully-reduced chunk it holds. After `n−1` more steps every rank
has all `n` reduced chunks.

```
steps: n−1     bytes sent per rank: (n−1)·S/n
cost:  (n−1)·α + (n−1)/n · S · β
```

### Total

```
T_ring_allreduce = 2(n−1)·α + 2(n−1)/n · S · β
```

### Why the bandwidth term is optimal

Every rank's contribution must reach every other rank, and the reduced result
must come back. Any all-reduce algorithm therefore requires each rank to
**send** at least `S(n−1)/n` bytes and **receive** at least `S(n−1)/n` bytes,
because a rank cannot know the sum without hearing from everyone, and cannot
inform everyone without speaking. Ring hits both bounds exactly:

```
lower bound on bytes per rank = 2(n−1)S/n     ← ring achieves this
```

So the `busbw` correction factor for all-reduce is `2(n−1)/n`. That is where
the factor in the glossary's `busbw` table comes from — it is not a fudge, it
is the optimal byte count.

### What ring is bad at

The latency term is `2(n−1)·α` — **linear in `n`**. At `n = 1024` and
`α = 5 µs`, that is 10.2 ms of pure latency before a single useful byte
moves. For small gradients this dominates completely, which is the entire
motivation for trees.

## The decomposition identity

```
AllReduce = ReduceScatter ∘ AllGather
```

Phase 1 *is* a reduce-scatter; phase 2 *is* an all-gather. This is not an
analogy — ring all-reduce literally executes them in sequence. Two direct
consequences used elsewhere in this plugin:

- **ZeRO-1/2 are bandwidth-free** relative to DDP: replacing
  `AllReduce(grads)` with `ReduceScatter(grads)` + `AllGather(params)` moves
  the same `2(n−1)S/n` bytes.
  (`parallelism-strategies/references/data-parallel-and-zero.md`)
- **Megatron sequence parallelism is bandwidth-free** relative to TP alone,
  for the same reason.
  (`parallelism-strategies/references/sequence-context-parallel.md`)

Any claim that one of these "reduces communication" is wrong. They reduce
*memory* at constant communication.

## Double binary trees

A binary tree broadcast takes `log₂(n)` steps rather than `n−1`, so the
latency term becomes logarithmic. The problem is bandwidth: in a binary tree
roughly half the ranks are leaves, and a leaf only ever receives — its
outbound link is idle.

NCCL's construction: because at least half the ranks in a binary tree are
leaves, you can build a **second** tree using the first tree's leaves as
internal nodes and its internal nodes as leaves. At most one rank is a leaf in
both, and no rank is internal in both. Send half the data down tree A and
half down tree B; now every rank is sending and receiving on both trees
combined, at full duplex rate.

```
T_tree_allreduce ≈ 2·log₂(n)·α  +  (bandwidth term comparable to ring)
```

| `n` | ring latency term | tree latency term | ratio |
|---|---|---|---|
| 8 | `14α` | `6α` | 2.3× |
| 64 | `126α` | `12α` | 10.5× |
| 1024 | `2046α` | `20α` | 102× |
| 16384 | `32766α` | `28α` | 1170× |

NCCL picks per-collective based on size and world size: trees for small and
medium messages where `α` dominates, rings for large messages where the
bandwidth term dominates and ring's optimality wins. `NCCL_ALGO` overrides the
choice; `NCCL_PROTO` selects the wire protocol (`LL`, `LL128`, `Simple`),
which trades a lower-latency but lower-bandwidth encoding for small messages.

Source: NVIDIA, "Massively Scale Your Deep Learning Training with NCCL 2.4" —
https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4

## Cost table for the collectives you will meet

`S` = bytes of *user* data per rank. `f(n)` = the `busbw` correction.

| Collective | Optimal bytes/rank | `f(n)` | Latency (ring) | Latency (tree) | Used by |
|---|---|---|---|---|---|
| all-reduce | `2(n−1)S/n` | `2(n−1)/n` | `2(n−1)α` | `2log₂(n)α` | DDP grads, TP activations |
| reduce-scatter | `(n−1)S/n` | `(n−1)/n` | `(n−1)α` | `log₂(n)α` | ZeRO grads, SP |
| all-gather | `(n−1)S/n` | `(n−1)/n` | `(n−1)α` | `log₂(n)α` | ZeRO-3 params, SP |
| broadcast | `S` | `1` | `(n−1)α` | `log₂(n)α` | init, param sync |
| reduce | `S` | `1` | `(n−1)α` | `log₂(n)α` | metric aggregation |
| all-to-all | `(n−1)S/n` | `(n−1)/n` | `(n−1)α` | — (no tree form) | MoE dispatch/combine |
| barrier | 0 | — | `log₂(n)α` | `log₂(n)α` | debugging, sync points |

**All-to-all deserves a warning.** It has no ring or tree structure to
exploit: every rank sends a distinct message to every other rank, so on a
hierarchical fabric it is the collective most exposed to topology. `n²`
messages, each potentially small — usually latency-bound at scale. This is why
MoE expert parallelism (`parallelism-strategies/references/expert-parallel-moe.md`)
is more topology-sensitive than dense training, and why hierarchical
all-to-all implementations (aggregate within the node, then exchange between
nodes) matter so much.

## Point-to-point

Pipeline parallelism uses `send`/`recv` rather than collectives. Cost is
simply `α + β·S` on the link between the two ranks. Its danger is not
bandwidth but **deadlock**: if two ranks both block on `recv` before either
`send`s, neither proceeds. Frameworks use batched/paired p2p or non-blocking
primitives to avoid this, which is why hand-rolled pipeline code hangs so
readily.

## Reading `nccl-tests` output

`nccl-tests` prints, per message size, both `algbw` and `busbw`:

```
algbw = S / T                  what the application sees
busbw = algbw · f(n)           what the wire carries
```

The rules for interpretation:

- Compare **`busbw`** to the link's theoretical bandwidth. It should approach
  it for large messages. `algbw` will not, and that is correct behaviour, not
  a fault.
- At small sizes both collapse — that is the `α` term, not a broken fabric.
  Read the *large-message asymptote* to judge bandwidth health.
- Convert the vendor's Gb/s to GB/s before comparing (divide by 8).
- Run the size sweep, not one size. The shape of `busbw` vs `S` tells you
  where `S* = α/β` sits on *your* fabric, which is the number that determines
  bucket sizes and FSDP wrapping granularity.

> Platform recipe: running `nccl-tests` on a specific cluster, and the
> interface/HCA settings it needs there, is skill `mlops:forge-train`.

## Worked example: is my all-reduce healthy?

`n = 16`, `S = 500 MB` of bf16 gradients, measured `T = 62 ms`.

```
algbw = 0.5 GB / 0.062 s          = 8.06 GB/s
busbw = 8.06 · 2(16−1)/16         = 8.06 · 1.875 = 15.1 GB/s
```

If those 16 ranks span two nodes over 200 Gb/s HDR InfiniBand — 25 GB/s —
then 15.1 GB/s is ~60% of line rate: plausible but leaving something on the
table. If instead all 16 were NVLink-connected (hundreds of GB/s), 15.1 GB/s
would indicate the collective is not using NVLink at all. **The same `algbw`
means completely different things depending on the topology**, which is why
`references/bandwidth-and-topology.md` is the companion to this file.
