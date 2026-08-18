# Expert Parallelism and MoE

Symbols: router → `notation-and-glossary.md`. `e` is the expert-parallel
degree; `E` is the number of experts; `k` is the top-k routing width.

## The asymmetry that defines MoE

A Mixture-of-Experts layer replaces one FFN with `E` FFNs and a router that
sends each token to `k` of them. Two parameter counts result, and conflating
them is the single most common MoE error:

| | Definition | Governs |
|---|---|---|
| `N_total` | every expert's parameters | **memory** — all of them must live somewhere |
| `N_active` | parameters touched per token ≈ `N_total · k/E` (for the expert part) | **FLOPs** — the `6N` rule uses this |

DeepSeek-V3 (arXiv:2412.19437) is the clean example: **671B total parameters,
37B activated per token**, trained on 14.8T tokens for 2.788M H800 GPU-hours.
An 18× gap between the memory number and the compute number.

Consequences that follow immediately:

- **MFU must use `N_active`.** Computing MFU with `N_total` inflates it by
  `E/k`. See `training-metrics/references/flop-counting-and-mfu.md`.
- **Memory planning must use `N_total`.** A "37B active" model needs 671B
  parameters' worth of state sharded somewhere.
- **Scaling-law comparisons need care.** An MoE at fixed `N_active` is not the
  same point as a dense model at that `N_active`.

## Expert parallelism

Since experts are independent FFNs, the natural split is to place `E/e`
experts on each of `e` ranks. Then:

```
per-rank expert weights = N_expert_total / e
```

But a token routed to expert `j` must physically reach the rank holding `j`.
That is an **all-to-all**:

```
1. router picks top-k experts per token
2. all-to-all #1: dispatch tokens to the ranks owning their experts
3. each rank runs its local experts on the tokens it received
4. all-to-all #2: combine — send results back to the tokens' owning ranks
```

Two all-to-all collectives per MoE layer per forward (and their transposes in
backward). Volume depends on routing, not on a fixed formula: if routing is
uniform, each rank sends `(1 − 1/e)` of its `k·tokens·h` elements.

All-to-all is the least forgiving collective on a hierarchical fabric — it
has no ring or tree structure to exploit and every rank talks to every other.
This is why `e` inside the node is strongly preferred, and why MoE runs are
more sensitive to network topology than dense runs of the same active size.
See `communication-backends/references/bandwidth-and-topology.md`.

## Load imbalance is the central engineering problem

Routing is learned, so nothing prevents the model from sending most tokens to
a few experts. Three mechanisms interact:

**Capacity factor.** Each expert accepts at most
`C = capacity_factor · (tokens · k / E)` tokens. Tokens beyond that are
**dropped** — they skip the expert entirely and pass through the residual.
`capacity_factor = 1.0` means zero slack and heavy dropping;
`1.25`–`2.0` is typical. Higher capacity costs memory and compute for padding
that may go unused.

**Auxiliary load-balancing loss.** An added term penalizing the correlation
between the fraction of tokens routed to an expert and the router's mean
probability for it. It works, but it is a *second objective* fighting the
language-modeling loss; too large a coefficient measurably costs quality.

**Expert-choice routing** (arXiv:2202.09368) inverts the assignment: instead
of each token choosing `k` experts, each **expert chooses its top-`C` tokens**.
Load is then perfectly balanced by construction — no capacity factor, no
dropping, no auxiliary loss. The cost is that a token may be picked by a
variable number of experts (including zero), which is fine for training but
awkward for autoregressive decoding, where future tokens are not available to
compete.

Newer large MoEs (DeepSeek-V3 among them) instead use auxiliary-loss-free
balancing: a per-expert bias added to the routing logits, adjusted online
based on observed load, which steers assignment without adding a gradient
term that competes with the loss.

## Diagnosing MoE-specific problems

| Symptom | Likely cause | Check |
|---|---|---|
| Throughput much worse than the dense-equivalent estimate | all-to-all is bandwidth-bound | is `e` crossing the node boundary? |
| Step time varies a lot step-to-step | routing imbalance changes the all-to-all volume per step | log per-expert token counts |
| Loss plateaus above the dense baseline | expert collapse — a few experts get everything | per-expert utilization histogram |
| Quality drops when you raise the aux-loss coefficient | balancing objective fighting the LM objective | sweep the coefficient, or move to expert-choice / bias-based balancing |
| Many tokens dropped | capacity factor too low for the current imbalance | log drop rate; it should be a fraction of a percent, not double digits |
| MFU looks implausibly high | FLOPs computed from `N_total` | recompute with `N_active` |

Per-expert token-count logging is the single highest-value instrument for an
MoE run. Without it, imbalance is invisible until it shows up as unexplained
throughput variance.

## Composing EP with other axes

- **EP replaces TP for the expert weights.** Both shard the FFN; doing both to
  the same weights multiplies communication for little gain. TP is still used
  for the attention blocks and the shared/dense parts.
- **EP and DP.** Ranks in an EP group process different experts for the *same*
  tokens, so EP behaves as its own axis:
  `n = d · t · p · c · e`. Frameworks often express this as an
  "expert data parallel" group holding the non-expert parameters.
- **EP and PP.** Compatible; MoE layers are typically placed on every other
  layer, so stage balancing must account for MoE layers being heavier.

## Implementation notes worth knowing

**Tutel** (arXiv:2206.03382) is the reference for adaptive MoE at scale: it
switches parallelism strategy *per iteration* as routing skew changes, and
implements the all-to-all with a hierarchical algorithm that respects the
intra-node/inter-node split. The general lesson generalizes past Tutel — a
static parallelism plan is a poor fit for a workload whose communication
pattern is data-dependent.

**Gate precision.** Router logits are frequently computed in fp32 even in a
bf16 model. Top-k over near-ties is discontinuous, so a small numerical
difference flips the assignment and makes the run non-reproducible across
runs and ranks. This is cheap insurance.

## Sources

- DeepSeek-V3 Technical Report — arXiv:2412.19437
- Mixture-of-Experts with Expert Choice Routing — arXiv:2202.09368
- Tutel: Adaptive Mixture-of-Experts at Scale — arXiv:2206.03382
