# Throughput and Scaling Efficiency

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`.

## Report throughput in tokens per second

`tokens/s` is the only throughput unit that survives a configuration change.
`samples/s` changes when `s` changes; `steps/s` changes when `B` changes;
`iterations/s` is meaningless without both.

```
tokens/s        = B · s / T
tokens/s/GPU    = B · s / (T · n)      ← the number to compare across scales
```

Report `tokens/s/GPU` when comparing world sizes. Perfect strong scaling holds
it constant; the shape of its decline is the diagnosis.

## Strong vs weak scaling

| | Strong | Weak |
|---|---|---|
| What is fixed | global batch `B` | per-GPU work |
| What grows with `n` | nothing | `B ∝ n` |
| Ideal | `T ∝ 1/n` | `T` constant |
| Efficiency | `E = T(n_ref)·n_ref / (T(n)·n)` | `E = T(n_ref)/T(n)` |
| Compute-to-comm ratio | **falls** with `n` | constant |
| Who runs it | someone who wants a result sooner | vendor scaling charts |

Never report an efficiency without saying which. "8 GPUs gave 5.2×" is 65%
strong-scaling efficiency — a normal, expected result if those 8 GPUs span two
nodes and the per-GPU batch is small, and a poor one if they are on a single
NVLink node.

Weak scaling is the easier chart and the less honest one, because `B` is a
hyperparameter: past the critical batch size, doubling `B` stops halving the
number of steps to convergence, so "linear weak scaling" can coexist with no
reduction in time to a target loss.

## The step-time model

```
T = T_compute/n_eff  +  T_comm(n)  +  T_bubble(p)  +  T_straggler
```

| Term | Depends on | Owner |
|---|---|---|
| `T_compute` | per-GPU FLOPs, kernel efficiency | `distributed-train:gpu-architecture` |
| `T_comm(n)` | volume / busbw + message count × α | `distributed-train:communication-backends` |
| `T_bubble(p)` | `(p−1)/(m+p−1)` × stage time | `distributed-train:parallelism-strategies` |
| `T_straggler` | tail of the per-rank duration distribution | `distributed-train:communication-backends` |

`n_eff` is the number of ranks doing distinct work — `d` for data parallelism;
under TP the ranks in a group share one sample's work, so compute per rank
falls with `t` as well.

### Fitting it from three measurements

Measure `T` at three world sizes, e.g. `n ∈ {8, 16, 32}`, holding the model,
`s`, `b`, and `m` fixed (strong scaling: `B` grows with `d`, so hold `b·m`
fixed and let `B = b·m·d`).

Assume the simplest useful forms — compute scales as `1/n`, and communication
has a constant plus a `log n` latency component:

```
T(n) = A/n + C + E·log₂(n)
```

Three equations, three unknowns. Solve for `A` (perfectly parallel compute),
`C` (the constant serial/bandwidth term), and `E` (the latency growth).

**Worked.** Measured: `T(8) = 4.20 s`, `T(16) = 2.55 s`, `T(32) = 1.80 s`.

```
4.20 = A/8  + C + 3E
2.55 = A/16 + C + 4E
1.80 = A/32 + C + 5E

subtract:  1.65 = A/16 − E        →  A = 16(1.65 + E)
           0.75 = A/32 − E        →  A = 32(0.75 + E)

16(1.65 + E) = 32(0.75 + E)  →  26.4 + 16E = 24 + 32E  →  E = 0.15
A = 16(1.80) = 28.8
C = 4.20 − 3.60 − 0.45 = 0.15
```

So `T(n) = 28.8/n + 0.15 + 0.15·log₂(n)`.

**Predict `n = 128`:** `T = 0.225 + 0.15 + 1.05 = 1.425 s`.
Ideal strong scaling from `n=8` would give `4.20 · 8/128 = 0.2625 s`, so the
predicted efficiency at 128 is `0.2625/1.425 = 18%`. The model says the
latency term (`1.05 s`, 74% of the predicted step) will dominate — which is a
concrete, falsifiable claim, and it names the fix: reduce message count or
switch to tree algorithms, not add bandwidth.

**Use the prediction as a test.** Run `n = 128` and compare. A large miss
tells you which assumption failed:

| Miss | Likely wrong assumption |
|---|---|
| Measured ≫ predicted, variance high | `T_straggler` — absent from the model, grows with `n` |
| Measured ≫ predicted, variance low | the fabric got worse (crossed a spine, oversubscription) |
| Measured ≈ predicted | the model is adequate; act on the term that dominates |
| Measured ≪ predicted | `E` was overfit — three points is few; add a fourth |

Three points is the minimum, not the ideal. Four or five make `E` much more
trustworthy.

## Diagnosis order when scaling is poor

1. **Confirm the measurement.** Warm-up excluded, several steps, a stated
   percentile. `references/benchmarking-methodology.md`.
2. **Strong or weak?** They have different expectations.
3. **Where did efficiency drop?** Plot `tokens/s/GPU` against `n`. A cliff at
   a specific `n` is a topology boundary (node, leaf switch); a smooth decline
   is the compute-to-comm ratio.
4. **Compute the ratio.** `6·b·m·s / bytes_per_elem` FLOPs per wire byte
   (router → `distributed-training-router/references/why-scaling-is-hard.md`). If it is small, no tuning
   will save it — raise `m`, `b`, or `s`.
5. **Is the step time variable?** Stable → bandwidth. Variable → straggler or
   bubble. This is the router's separating test, applied to a scaling curve.
6. **Only now profile.** `distributed-train:gpu-architecture` →
   `gpu-architecture/references/profiling-with-nsight.md`.

## Common patterns

| Curve | Reading |
|---|---|
| ~linear to 8, cliff at 16 | node boundary crossed; check which parallelism axis crossed it |
| Smooth decline from the start | compute-to-comm ratio too low — per-GPU batch too small |
| Flat then sudden collapse at large `n` | latency term (`2(n−1)α` for ring) taking over, or a spine boundary |
| Good scaling but time-to-loss unchanged | `B` past the critical batch size — throughput is not progress |
| Efficiency > 100% | usually a cache effect from a smaller per-GPU working set, or the baseline was measured badly |
| Highly variable between identical runs | placement varies, or a straggler that appears intermittently |

The fourth row is the trap that matters most. **Throughput is not progress.**
A configuration that doubles `tokens/s` by doubling `B` past the critical
batch size does not halve time to a target loss. Always confirm a throughput
win survives as a time-to-loss win before adopting it —
`references/compute-budget-and-scaling-laws.md`.

## What to hold fixed when comparing configurations

Changing the parallelism changes several things at once. To attribute a
throughput difference, hold these constant:

| Hold fixed | Why |
|---|---|
| `B = b·m·d` | otherwise it is a different experiment, not a faster one |
| `s` | changes FLOPs per sample and the attention fraction |
| dtype | changes both peak and wire bytes |
| recomputation setting | changes HFU but not MFU — and changes step time a lot |
| dataset order and seed | otherwise loss comparison is meaningless |
| the measured steps | same range, same warm-up exclusion |

If a comparison changes `B`, report tokens/s **and** the loss curve. A
throughput number alone cannot distinguish "faster" from "doing less useful
work per token".
