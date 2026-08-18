# Composing N-D Parallelism

Putting `d`, `t`, `p`, `c` (and `e`) together into one mesh. Symbols: router →
`notation-and-glossary.md`.

## The two invariants, restated because everything below checks against them

```
n = d · t · p · c            every rank has one coordinate per axis
B = b · m · d                t, p, c do NOT multiply the batch
```

A config that violates the first will not launch. A config that violates the
second silently trains on a different batch size than intended and produces a
different loss curve — the most expensive kind of error, because it looks
like it worked.

## The ordering, derived from volume

Per step, per rank, in elements (from the SKILL.md comm-volume table):

| Axis | Volume per step | On the critical path? | Overlappable? |
|---|---|---|---|
| TP | `4 · L · b · s · h` | yes — between two matmuls of one layer | barely |
| CP | `2 · b · s · h · (c−1)/c` per attention | partly | yes, with attention compute |
| PP | `2 · m · b · s · h` per boundary | it *is* the serialization | n/a |
| DP | `2Ψ` (ZeRO-1/2) or `3Ψ` (ZeRO-3) | no | yes, with backward |

Plug in a 70B-class shape — `L=80`, `h=8192`, `s=8192`, `b=1`, `Ψ=70e9`,
bf16 (2 bytes):

```
TP:  4 · 80 · 1 · 8192 · 8192 · 2 B   = 43.0 GB   per step, critical path
CP:  2 · 1 · 8192 · 8192 · 2 B        = 0.27 GB   per attention, overlappable
PP:  2 · 1 · 8192 · 8192 · 2 B        = 0.27 GB   per micro-batch per boundary
DP:  2 · 70e9 · 2 B                   = 280 GB    per step, overlappable
```

DP's raw number is the largest, but it happens **once** per step and hides
under the entire backward pass. TP's 43 GB happens in `4L = 320` separate
blocking collectives, each of which stalls the SMs. Divide by the link each
would run on:

| Axis | Volume | On NVLink (~450 GB/s eff.) | On 400G IB (50 GB/s) |
|---|---|---|---|
| TP | 43.0 GB | 96 ms, unhidden | 860 ms, unhidden |
| DP | 280 GB | 622 ms, hidden | 5.6 s, mostly hidden |
| PP | 0.27 GB/boundary | 0.6 ms | 5.4 ms |

TP on the fabric costs ~860 ms of pure stall against a step that should take
a few hundred ms. TP on NVLink costs 96 ms that partly overlaps. PP is
negligible on either. Hence:

```
TP  innermost  (fastest link, NVLink, t ≤ node size)
CP  next       (overlappable, still prefers fast links)
PP  across nodes (tiny volume, tolerates latency)
DP  outermost  (large volume but fully overlappable)
```

This is not a convention — it falls out of the table. Llama 3 405B states the
same order, `[TP, CP, PP, DP]` (arXiv:2407.21783).

## Worked configuration 1 — 70B dense, 64× H100 (8 nodes × 8)

Constraints: `n = 64`. 80 GB per GPU. `s = 8192`. Target `B = 512` samples.

**Step 1 — will ZeRO-2 alone do it?** Model states are `16Ψ = 1120 GB`.
ZeRO-2 gives `2Ψ + (2Ψ+12Ψ)/d`. With `d = 64`: `140 + 15.3 = 155 GB` per
rank. Over 80 GB. **No.**

**Step 2 — ZeRO-3?** `16Ψ/64 = 17.5 GB` per rank. Fits, with ~62 GB left for
activations. Now check activations. Per layer, no model parallelism, using
`s·b·h·(34 + 5as/h)` with `h=8192`, `a=64`, `b=1`, `s=8192`:

```
5as/h = 5·64·8192/8192 = 320
per layer = 8192·1·8192·(34+320) = 2.38e10 bytes = 23.8 GB     ← per layer!
```

80 layers of that is impossible. Activations, not states, are the binding
constraint. **Recomputation and/or TP are mandatory.**

**Step 3 — TP=8 with SP, inside the node.** TP+SP divides the whole per-layer
figure by `t`: `23.8/8 = 2.98 GB` per layer. Still 238 GB over 80 layers.
Add selective recomputation, which removes the `5as/h` term
(arXiv:2205.05198): `(8192·1·8192·34)/8 = 0.29 GB` per layer → **22.8 GB**
for all 80 layers. Now it fits.

**Step 4 — fill the mesh.** `t = 8` consumes one node.
`c = 1` (8192 is not long enough to need CP).
`p = 1` (the model fits with TP+ZeRO on the remaining axis).
`d = n/(t·p·c) = 64/8 = 8`.

**Step 5 — batch.** `B = b·m·d = 1 · m · 8 = 512` → `m = 64`. With `p = 1`
there is no bubble, so a large `m` is pure benefit: it raises the
compute-to-comm ratio `6·b·m·s/bytes` and lets DP's `3Ψ` hide.

```
Config 1:  (d, t, p, c) = (8, 8, 1, 1),  n = 64 ✓
           b = 1, m = 64, d = 8  →  B = 512 ✓
           ZeRO-3/FSDP over the d axis, TP=8 + SP inside each node,
           selective activation recomputation.
```

**Rejected alternatives, with the arithmetic:**
- `t = 16` (across two nodes): TP's 43 GB/step crosses IB → ~860 ms of stall.
  Rejected.
- `p = 8, t = 1`: needs `m ≥ 32` for a <20% bubble, which is fine, but with
  `t=1` the per-layer activation term is 23.8 GB and a stage holding 10 layers
  with `p` micro-batches in flight is far over 80 GB even with recompute.
  Rejected on capacity.
- `t = 4, p = 2`: works, but PP buys nothing here (the model already fits) and
  costs a bubble. Rejected on serialization.

## Worked configuration 2 — 405B dense, 1024× H100 (128 nodes × 8), `s = 8192`

Constraints: `n = 1024`, target `B = 2048`.

Model states: `16 × 405e9 = 6.48 TB`. Even at `d = 1024`, ZeRO-3 gives 6.3 GB
per rank of states — trivially fine. The problem is again activations, and
also that a *single layer's* weights must fit while gathered.

**TP.** `t = 8`, in-node, with SP. Divides both weights and activations by 8.

**PP.** 126 layers at `h = 16384` — even with TP=8, keeping all layers'
gathered parameters and in-flight activations on one node's worth of rank is
tight, and ZeRO-3's `3Ψ` traffic across 128 nodes is large. `p = 16` gives
~8 layers per stage.

**CP.** `c = 1` at `s = 8192`.

**DP.** `d = 1024/(8 · 16 · 1) = 8`.

**Batch and bubble.** `B = b·m·d = 2048` with `d = 8` → `b·m = 256`. Take
`b = 1`, `m = 256`. Bubble with 1F1B at `p = 16`, `m = 256`:
`(16−1)/(256+16−1) = 5.5%`. With interleaving at `v = 2`: 
`(1/2)·15/256 = 2.9%`. Acceptable.

```
Config 2:  (d, t, p, c) = (8, 8, 16, 1)
           check: d·t·p·c = 8 · 8 · 16 · 1 = 1024 = n ✓
           b = 1, m = 256, d = 8  →  B = 1 · 256 · 8 = 2048 ✓
           1F1B interleaved v = 2  →  bubble 2.9%
```

Evaluate `d·t·p·c` as arithmetic every time. The common failure is quoting a
tuple that "looks like" 1024 — `(8, 16, 8, 1)` also multiplies to 1024 but
puts TP across two nodes, which the volume table already rejected.

## Worked configuration 3 — 405B at 131k context, 1024× H100

Same model, `s = 131072`. The `5as/h` activation term scales with `s`, and
attention's FLOPs scale with `s²`, so the previous config's assumptions all
break.

**CP is now mandatory.** At `s = 131072`, `h = 16384`, `a = 128`:
`5as/h = 5·128·131072/16384 = 5120`, dwarfing the `34`. The attention term
is ~99% of activation memory. Only CP shards it across ranks *and* splits the
`O(s²)` attention work.

**Budget the intra-node axes.** `t · c ≤ 8` if both must stay in-node. Take
`t = 8, c = 1`? No — `c = 1` leaves `s = 131072` on one rank. Take
`t = 4, c = 2`? Still only 65k per rank. Llama 3's answer at this shape is to
let CP take a larger degree and accept crossing the node boundary for the
ring, because the ring's per-step message is small and overlappable, unlike
TP's all-reduce.

```
Config 3:  (d, t, p, c) = (2, 8, 8, 8),  n = 2·8·8·8 = 1024 ✓
           b = 1, m = 128, d = 2  →  B = 256 samples = 33.6M tokens ✓
```

`B` in *samples* is small because each sample is 131k tokens; the token count
per step is what matters for optimization. Expect MFU to drop — Llama 3
reports **38%** at 131k context versus 43% at 8k, because CP's rolling K/V
exchange has less compute to hide behind than FSDP's single all-reduce
(arXiv:2407.21783).

## Degrees of freedom — how many knobs there really are

Given `n` and a target `B`, the free choices are `(t, p, c, b, m)`; then
`d = n/(t·p·c)` and the batch invariant fixes the last one. Practical bounds
collapse the space fast:

| Knob | Bound | Reason |
|---|---|---|
| `t` | `≤` node size, divides `a` and `a_kv` | critical-path all-reduce, head splitting |
| `c` | `≥ s / s_max_per_rank` | the reason CP exists |
| `p` | smallest that makes it fit | bubble grows with `p` |
| `m` | `≥ 4p` | bubble `(p−1)/(m+p−1)` |
| `b` | as large as memory allows after the above | tile efficiency, fewer launches |
| `d` | determined | `n/(t·p·c)` |

That usually leaves two or three viable configs, which is few enough to
benchmark rather than argue about. `training-metrics/references/benchmarking-methodology.md`
has the protocol for comparing them honestly.

## Published configurations, for calibration

| Run | Mesh | Reported | Source |
|---|---|---|---|
| Megatron PTD-P, 1T params | `t=8, p=64, d=6` on 3072 A100 | 163 TFLOP/s/GPU, 52% of peak | arXiv:2104.04473 |
| PaLM 540B | 2-way pod-level data parallel over 6144 TPU v4 | 46.2% MFU, 57.8% HFU | arXiv:2204.02311 |
| MegaScale 175B | 12288 GPUs | 55.2% MFU | arXiv:2402.15627 |
| Llama 3 405B | 4D `[TP, CP, PP, DP]`, up to 16384 H100 | 38–43% BF16 MFU | arXiv:2407.21783 |

Two patterns hold across all of them: `t ≤ 8`, and `p` in the range that
keeps `m ≥ 4p` at the chosen `B`. A proposed config that violates either
deserves a second look.

One caveat on the MegaScale row: its 55.2% is measured with a modified block
structure and sliding-window attention, so it is not an apples-to-apples MFU
against the others —
`training-metrics/references/flop-counting-and-mfu.md` under comparing MFU
across architectures. The same run is also the clearest published case of `m`
being raised by increasing `B` through the optimizer rather than by giving up
`d`; see `references/pipeline-parallel.md`.

## Checklist before committing a config

1. `n = d·t·p·c` — evaluate it, do not eyeball it.
2. `B = b·m·d` equals the intended global batch. If you changed `t`, `p`, or
   `c` and `B` moved, the change is a *different experiment*.
3. `m ≥ 4p`, or accept the bubble in writing.
4. `t` divides `a` and `a_kv`; `t` ≤ node size.
5. Per-rank memory estimated with the activation formula, not just states.
6. State which budget each axis is buying (capacity / bandwidth /
   serialization) and what it costs in the others.

> Platform recipe: mapping this mesh onto real nodes — how many nodes to
> request, launch topology, rank placement — is skill `mlops:forge-train`.
