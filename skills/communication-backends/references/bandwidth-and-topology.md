# Bandwidth and Topology

Where the bytes actually travel, and why the same collective costs wildly
different amounts depending on which links it lands on. Symbols: router →
`distributed-training-router/references/notation-and-glossary.md`.

## The hierarchy

Per GPU, rounded. Note the units: vendors quote network in **Gb/s**, memory
in **GB/s**.

| Level | Bandwidth | vs HBM | Latency scale |
|---|---|---|---|
| Registers | ~100 TB/s aggregate | ~30× | ~1 cycle |
| Shared memory / L1 | ~10–20 TB/s aggregate | ~5× | ~30 cycles |
| L2 cache | several TB/s | ~2× | ~200 cycles |
| HBM3 (H100 SXM) | 3.35 TB/s | 1× | ~500 cycles |
| HBM3e (B200) | 8 TB/s | 2.4× | ~500 cycles |
| NVLink 4 (H100, per GPU, bidir.) | 900 GB/s | ~1/4 | ~1–2 µs |
| NVLink 5 (B200, per GPU, bidir.) | 1.8 TB/s | ~1/4.4 | ~1–2 µs |
| PCIe Gen5 x16 | ~64 GB/s bidir. | ~1/50 | ~1–2 µs |
| NDR InfiniBand (400 Gb/s) | 50 GB/s | ~1/67 | ~2–5 µs |
| HDR InfiniBand (200 Gb/s) | 25 GB/s | ~1/134 | ~2–5 µs |
| 100 GbE | 12.5 GB/s | ~1/268 | ~10–50 µs |
| NVMe SSD | 3–7 GB/s | ~1/500 | ~10–100 µs |

(GPU figures from NVIDIA's published specifications: H100 SXM 80 GB @
3.35 TB/s, NVLink 900 GB/s; DGX B200 quotes 1,440 GB across 8 GPUs at
64 TB/s aggregate — 180 GB and 8 TB/s per GPU — with 14.4 TB/s aggregate
NVLink, i.e. 1.8 TB/s per GPU.)

Two structural facts drive every placement decision:

1. **The intra-node → inter-node step is the cliff.** NVLink to InfiniBand is
   a factor of 18–36×. Any traffic that must cross it should be small,
   infrequent, or overlappable. That is exactly the TP-inside / PP-and-DP-
   outside ordering in
   `parallelism-strategies/references/composing-nd-parallelism.md`.
2. **NICs are shared.** A node with 8 GPUs and 4× 400 Gb/s NICs has 200 GB/s
   of egress for 8 GPUs — 25 GB/s each *if perfectly balanced*. Compare that
   to 900 GB/s of NVLink per GPU.

## Intra-node topologies

| Topology | Shape | Consequence |
|---|---|---|
| Full NVLink mesh / NVSwitch | every GPU to every other at full rate | any `t ≤ 8` placement works; all-to-all is cheap |
| Partial NVLink (pairs/quads) | some pairs on NVLink, others over PCIe | `t = 2` or `4` fine, `t = 8` hits PCIe; rank ordering matters |
| PCIe-only | all GPU-GPU via PCIe root complex | TP is generally not viable; prefer ZeRO/PP |
| PCIe with GPUDirect P2P disabled | traffic staged through host memory | catastrophically slow; shows as a mysterious 5–10× collective slowdown |

The practical check is a device-to-device bandwidth matrix. A healthy
8-GPU NVSwitch node shows near-uniform high bandwidth between all pairs; a
partial-NVLink node shows a visible block structure. Diagnosing "why is
`t=8` slow but `t=4` fine" almost always ends at that matrix.

## Inter-node topologies

**Fat tree.** The standard HPC/AI fabric: leaf switches connect nodes, spine
switches connect leaves. Its key parameter is the **oversubscription ratio** —
if leaf switches have 32 downlinks and 16 uplinks, that is 2:1, and traffic
between nodes on different leaves gets half the bandwidth of traffic within a
leaf. All-reduce that crosses the spine on an oversubscribed fabric runs at
the spine's share, not the NIC's rate.

**Rail-optimized.** Each GPU `i` in every node connects to switch (rail) `i`.
A collective in which every rank talks only to the *same-indexed* rank in
other nodes stays within one rail and never touches the spine. This is why
NCCL's ring construction tries to keep ranks rail-aligned, and why a
misordered rank-to-GPU mapping can cost a large fraction of fabric bandwidth
with no error message.

Rail alignment has a second motivation that is usually left implicit:
**ECMP hashing**. Multi-path fabrics spread flows across uplinks by hashing
the flow's headers, which works when there are many small flows and fails when
there are a few enormous ones — two elephant flows landing on the same uplink
halve each other while a parallel link sits idle. Training traffic is exactly
the bad case: few flows, all large, all simultaneous. Keeping a collective
inside one rail avoids the hash entirely.

The scheduling consequence is the practical one. If a set of leaf switches
covers some number of hosts, a job placed *within* one such set never crosses
the spine for its heaviest collectives. Placement is therefore a bandwidth
decision, not just a capacity decision — and it is why the same job can be
measurably faster or slower depending on where the scheduler put it, with no
configuration difference at all.

**Dragonfly / torus.** Lower diameter for a given radix, but the routing is
topology-aware and adversarial traffic patterns (all-to-all) can create
hotspots. Less common in AI clusters than fat trees.

## What a production fabric tunes beyond the topology

Worth knowing exists, because these are the levers a cluster team pulls when
the topology is already right and collectives are still slow or unstable
(MegaScale, arXiv:2402.15627):

| Lever | Why the default is not enough |
|---|---|
| **Congestion control** | ECN-marking schemes alone (DCQCN-style) react to queue buildup, but tuning them across thousands of flows is fragile; production systems combine ECN signals with direct RTT measurement (Swift-style) to get both a precise congestion signal and a fast response |
| **Retransmit timers** | defaults are tuned for links that fail rarely; on a fabric where a flapping link is a weekly event, a shorter, adaptive retransmit turns a stall into a hiccup |
| **Port splitting** | splitting a high-rate downlink into two lower-rate ports increases the number of paths the ECMP hash can choose from, reducing collision probability |
| **Cable and transceiver QC** | link flapping usually traces to signal quality between NIC, cable, and switch port — no software setting fixes a marginal transceiver |

The last row is the one to remember when a job intermittently loses
throughput and every software knob has been tried: **at scale, some fraction
of the fabric is always physically marginal**, and finding it is a hardware
process, not a debugging session. That is what the node self-check suite in
`references/fault-tolerance-and-node-diagnostics.md` exists to automate.

> Platform recipe: which of these are configured on a given cluster, and what
> values they hold, is `mlops:forge-train`. This table explains what class of
> problem each addresses.

## Hierarchical collectives

Because intra-node bandwidth exceeds inter-node bandwidth by an order of
magnitude, the right algorithm for a multi-node collective is usually
two-level:

```
all-reduce over 8 nodes × 8 GPUs:
  1. reduce-scatter within each node   (NVLink, cheap)
  2. all-reduce across nodes, on 1/8 of the data each  (fabric, the expensive part)
  3. all-gather within each node       (NVLink, cheap)
```

Step 2 moves `S/8` per rank instead of `S`. NCCL does this automatically for
multi-node collectives; the reason to know it is that it explains why the
*inter-node* portion is the only part worth optimizing, and why an
8-GPU-per-node job and a 1-GPU-per-node job with the same world size behave
completely differently.

For all-to-all (MoE) the same idea applies but is harder: aggregate the
messages destined for each remote node into one large message, exchange, then
scatter locally. Implementations that skip this send `n²` small messages
across the fabric — see Tutel (arXiv:2206.03382).

## Computing what you should expect

Given a config, predict the communication time before measuring it:

```
1. bytes per step per rank      ← parallelism comm-volume table
2. the link that traffic uses   ← from the mesh ordering
3. busbw correction f(n)        ← per collective
4. T_comm = bytes / (busbw_link) + n_messages · α
```

Worked: 70B model, ZeRO-3 over `d = 8` ranks spanning 8 nodes, bf16.

```
volume  = 3Ψ = 3 · 70e9 elements = 420e9 bf16 elements = 840 GB
link    = 400 Gb/s NDR = 50 GB/s per node
         (hierarchical: the inter-node part carries ~ (d−1)/d of it)
T_comm ≈ 840 GB · (7/8) / 50 GB/s ≈ 14.7 s per step
```

If the compute for that step is ~2 s, the job is hopelessly bandwidth-bound
and the answer is not "tune NCCL" — it is to change the parallelism so the
`3Ψ` does not cross the fabric, or to raise `m` so a step covers more tokens.
That is the value of doing this arithmetic first: it distinguishes a
configuration error from a tuning opportunity.

## Where the node boundary shows up

| Observation | Reading |
|---|---|
| Scaling efficiency ~95% to 8 GPUs, ~60% at 16 | you crossed the node boundary; check what axis crossed it |
| `t = 8` slower than `t = 4` on the same node | partial NVLink, or a rank-ordering problem |
| Two nodes fine, eight nodes much worse | spine oversubscription, or rail misalignment |
| Performance varies between otherwise identical jobs | placement varies — the scheduler put you on different leaves |
| MoE much worse at multi-node than dense at the same size | all-to-all crossing the fabric |

The last row generalizes: **the collective's structure determines how badly
the topology hurts.** Ring degrades gracefully; all-to-all does not.

> Platform recipe: which interfaces, HCAs, or rails exist on a given cluster,
> and how to pin them, is skill `mlops:forge-train` → `error-patterns.md` §3.
> This file explains why those settings matter, not what to set.

## The one-line summary

Place the highest-volume, least-overlappable traffic on the fastest link. Every
parallelism ordering rule in this plugin is that sentence plus the table at the
top of this file.
