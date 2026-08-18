# Why Scaling Is Hard

Symbols: `references/notation-and-glossary.md`.

Scaling a training job is not "add GPUs, go faster". Four budgets — compute,
capacity, bandwidth, serialization — change at different rates as `n` grows,
and the one that binds changes with them. This file explains the mechanism
behind each curve so the sibling skills' numbers have somewhere to sit.

## 1. Amdahl sets a ceiling you cannot buy your way past

Split a step into a part that parallelizes perfectly and a part that does not:

```
T(n) = T_par/n + T_serial            speedup S(n) = T(1)/T(n)
```

With serial fraction `f = T_serial/T(1)`, the ceiling is `S(∞) = 1/f`. At
`f = 0.02` — two percent — you can never exceed 50×, no matter how many GPUs
you buy. At `n = 64` you already get 28.3×, a 44% efficiency.

In distributed training, `T_serial` is not one thing:

| Serial component | Grows with |
|---|---|
| gradient all-reduce that did not overlap | `n` (latency term) |
| pipeline bubble | `p` |
| optimizer step on unsharded state | constant, but never shrinks |
| straggler wait | `max` over `n` ranks — grows with `n` |
| checkpoint write | constant per write, more frequent at scale (see below) |

The straggler term is the nasty one: it is a **maximum over `n` samples**, so
even with identical hardware it grows like the tail of the per-rank duration
distribution. Doubling `n` does not double it, but it never shrinks either.

## 2. Strong scaling vs weak scaling

- **Strong scaling** — fix the global batch `B`, add GPUs. Per-GPU work
  `B/(d)` shrinks, communication per step stays roughly constant → the
  compute-to-communication ratio *falls*, efficiency falls. This is the hard
  case and the one people actually run when they want a result sooner.
- **Weak scaling** — grow `B` with `n`. Per-GPU work is constant, so
  efficiency is nearly flat. This is what vendor scaling charts show. It is
  also *not free*: `B` is a hyperparameter, and beyond the critical batch size
  more samples per step stop buying you fewer steps.

When someone reports "8 GPUs gave 5.2×", the first question is which of the
two they measured. See `training-metrics/references/throughput-and-scaling-efficiency.md`.

## 3. The compute-to-communication ratio is the real scaling number

For plain data parallelism, per step, per rank:

```
compute   ≈ 6 · Ψ · (b·m·s)   FLOPs        (6N per token; see training-metrics)
comm      ≈ 2 · Ψ · bytes_per_elem  bytes   (all-reduce, busbw-corrected)
```

Their ratio is the *arithmetic intensity of the whole step* with respect to
the network:

```
ratio = 6 · b · m · s / bytes_per_elem      [FLOPs per byte on the wire]
```

Note what is **not** in it: `Ψ`. Model size cancels. What buys scaling is
**tokens processed per parameter-update**, i.e. `b·m·s`. That single fact
explains most of practice:

- gradient accumulation improves scaling efficiency (raises `m`)
- long context improves it (raises `s`)
- tiny per-GPU batches destroy it — the classic strong-scaling wall
- bf16 gradients halve wire bytes and double the ratio

Compare `ratio` against the machine balance `P_peak / busbw` in FLOPs per
byte. If `ratio` is below it, you are bandwidth-bound and no kernel
optimization will help.

## 4. Capacity does not scale with `n` — until you shard

Adding GPUs adds HBM, but plain DDP **replicates** everything, so per-GPU
capacity pressure is flat at `(2 + 2 + K)·Ψ` bytes of model state regardless
of `n`. Only sharding — ZeRO stages, TP, PP — converts world size into
capacity. This is why the answer to "I got OOM, I'll add nodes" is *no,
adding nodes changes nothing under DDP*.

Once you shard, capacity scales as `1/d` (ZeRO-3), `1/t` (TP), `1/p` (PP) —
but each divisor is paid for in the bandwidth budget. The exchange rates are
in `parallelism-strategies/references/composing-nd-parallelism.md`.

## 5. Bandwidth is a hierarchy, and it is very steep

Rounded orders of magnitude, per GPU:

| Level | Bandwidth | Ratio to HBM |
|---|---|---|
| registers / SMEM | ~10–20 TB/s effective | ~5× |
| HBM (H100 SXM) | 3.35 TB/s | 1× |
| NVLink (H100, bidirectional) | 900 GB/s | ~1/4 |
| PCIe Gen5 x16 | ~64 GB/s bidirectional | ~1/50 |
| NDR InfiniBand, 400 Gb/s | 50 GB/s | ~1/67 |
| NVMe SSD | 3–7 GB/s | ~1/500 |

Each step down is a factor of several. Every parallelism decision is a
decision about *which level a given tensor crosses*, and the ordering rules
in `parallelism-strategies` are derived from exactly this table: the axis
with the most traffic per step goes on the fastest level.

## 6. Failures stop being exceptional

Treat per-GPU failures as independent with mean time between failures
`MTBF_1`. The job-level MTBF is:

```
MTBF_job ≈ MTBF_1 / n
```

At `MTBF_1 = 5` years and `n = 16384`, `MTBF_job ≈ 2.7 hours`. This is not
hypothetical: the Llama 3 405B report documents **466 job interruptions over
54 days** on up to 16K H100s — 419 unexpected, ~78% of those traced to
confirmed hardware issues, with GPU failures the single largest category at
58.7% of unexpected interruptions (arXiv:2407.21783).

Two consequences:

1. Checkpoint interval is an optimization problem, not a habit. The
   Young/Daly result gives `T_opt ≈ sqrt(2 · C · MTBF)` for checkpoint cost
   `C` — derived in `communication-backends/references/hang-and-straggler-debugging.md`
   (Daly, *Future Generation Computer Systems* 22(3), 2006, pp. 303–312).
2. Reproducibility must survive a restart. A run that cannot resume to the
   same loss curve cannot be debugged at this scale.

## 7. Why the binding budget moves

A rough map of what binds as you scale a fixed model:

| Regime | Binding budget | Symptom |
|---|---|---|
| 1 GPU, model doesn't fit | capacity | OOM before step 1 |
| 1 node, 8 GPUs, NVLink | compute | high MFU, scales nearly linearly |
| 2–8 nodes | bandwidth | MFU drops at the node boundary |
| tens of nodes, PP introduced | serialization | bubble appears in the trace |
| hundreds+ nodes | serialization + failures | step-time tail, restarts dominate |

The practical reading: **the fix that worked at the last scale is usually the
wrong fix at the next one.** Re-run the separating test in the router's
SKILL.md at each scale rather than carrying forward a diagnosis.

## 8. What good looks like

Published, verified reference points for "this is achievable":

| Run | Hardware | MFU | Source |
|---|---|---|---|
| PaLM 540B | 6144 TPU v4 | 46.2% MFU (57.8% HFU) | arXiv:2204.02311 |
| Megatron PTD-P, 1T params | 3072 A100 | 52% of peak, 163 TFLOP/s/GPU | arXiv:2104.04473 |
| MegaScale, 175B | 12288 GPUs | 55.2% MFU | arXiv:2402.15627 |
| Llama 3 405B | up to 16384 H100 | 38–43% BF16 MFU | arXiv:2407.21783 |

Note the direction: the largest, most carefully engineered runs sit in the
high 30s to mid 50s. A 25% MFU is not automatically broken, and a claim of
70%+ on a dense transformer deserves an audit of how the FLOPs were counted
— see `training-metrics/references/flop-counting-and-mfu.md`.
