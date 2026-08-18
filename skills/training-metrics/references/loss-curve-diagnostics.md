# Loss Curve Diagnostics

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`.

Two categories of problem, and they need different first moves:

- **Correctness** — the loss changed and it should not have. Run the checklist
  below *first*. A correctness bug looks exactly like a hyperparameter problem
  and wastes far more time.
- **Optimization** — spikes, plateaus, divergence in a run that is otherwise
  computing the right thing.

## Correctness: "the loss changed when I changed the parallelism config"

Parallelism must not change the mathematics. If loss moved, one of these is
the reason. Check in order — the first is the cause about half the time.

**1. Is `B = b·m·d` identical?**

`t`, `p`, and `c` do **not** multiply the batch. Ranks inside a tensor,
pipeline, or context group process the *same* samples. So going from
`(d,t,p,c) = (8,1,1,1)` to `(4,2,1,1)` at fixed `b` and `m` **halves** `B`.
The run is not "the same experiment, faster" — it is a different batch size
with a different effective learning rate.

Frameworks differ in whether the config's "batch size" means `b`, `B`, or
per-DP-group. Print `b`, `m`, `d`, and the computed `B` at startup, every run.

**2. Is the loss a token-mean or a mean of per-rank means?**

With unequal token counts per rank (padding, variable-length sequences,
dropped MoE tokens), averaging per-rank mean losses is biased toward ranks
with fewer tokens:

```
wrong:   mean_over_ranks( sum_loss_r / count_r )
right:   all_reduce(sum_loss) / all_reduce(count)
```

Reduce the `(sum, count)` pair and divide once. The same applies to gradient
normalization — if gradients are normalized per rank by that rank's token
count and then averaged, the result is not the gradient of the global mean
loss.

**3. Is RNG seeded correctly per parallel axis?**

| Axis | Dropout mask must be |
|---|---|
| DP (`d`) | **different** per rank — they see different samples |
| TP (`t`) | **identical** across the group — they compute one sample's layer together |
| PP (`p`) | independent per stage (different layers) |
| CP (`c`) | identical for shared state; per-position for the sharded part |

A common bug is seeding by global rank everywhere, which desynchronizes
dropout within a TP group and silently computes a different function than the
single-GPU model.

**4. Is the data sharded without overlap or gaps?**

Each sample should be seen once per epoch across all `d` ranks. Check that the
sampler's shard count is `d` and not `n`. With `n = d·t·p·c`, using `n`
produces `t·p·c` × the intended shards and each DP rank sees a fraction of its
data.

**5. Is the shuffle seed shared across ranks?**

The permutation must be identical across ranks so that shard `i` is
well-defined; only the *slice* differs. A per-rank shuffle seed means
overlapping and missing samples.

**6. Are unequal dataloader lengths handled?**

If one rank runs out of batches first, it either hangs at the next collective
(`communication-backends/references/hang-and-straggler-debugging.md`) or, with
a drop-last mismatch, contributes a partial batch that skews the mean.

**7. Under PP, is the loss computed only on the last stage and broadcast?**

Logging a loss from a non-final stage yields garbage; averaging a loss that
only one stage has, across all stages, yields garbage divided by `p`.

### The reproducibility test

The strongest single check: run the new config and the old config for ~200
steps with the same seed and the same data order, and plot both losses. If
they are not nearly identical (a small numerical difference from different
reduction orders is fine; a visible offset is not), the change is a
correctness bug, not a performance change.

## Optimization: the spike taxonomy

| Shape | Typical cause | Distributed-specific? |
|---|---|---|
| Single spike, recovers in tens of steps | a hard batch, a long/duplicated document | no |
| Spike then divergence to a plateau | logit growth / attention entropy collapse | no |
| Spike exactly at a resume point | data order not restored, or optimizer state not restored | **yes** |
| Spike at a regular interval | a periodic data source (a shard boundary, an eval that mutates state) | maybe |
| Loss jumps on exactly one rank's contribution | a bad shard, or corrupt data on one node | **yes** |
| Loss steps down suddenly at a fixed step | learning-rate schedule boundary — not a spike | no |
| Loss becomes NaN | fp16 overflow, or a bad sample; check the loss scaler | maybe |
| Slow upward drift | learning rate too high for the current batch size, or weight decay misconfigured | no |
| Loss flat from step 0 | bf16 master weights swallowing updates; or lr ≈ 0; or a detached graph | maybe |

The **resume** row deserves emphasis because it is the most common
distributed-specific loss artifact. A correct resume restores model weights,
optimizer state (both moments), the learning-rate schedule position, the RNG
state, **and the dataloader position**. Missing the last one replays already-
seen data and produces a characteristic dip-then-spike.

## Instability mechanisms worth knowing

Two failure modes reproduce reliably at small scale with high learning rates
(arXiv:2309.14322, "Small-scale proxies for large-scale Transformer training
instabilities"):

- **Attention logit growth.** The pre-softmax logits grow without bound, the
  softmax saturates, gradients through attention vanish, and the loss plateaus
  or diverges. Mitigated by QK-layernorm (normalizing Q and K before the dot
  product).
- **Output logit divergence.** The output logits drift away from the log
  probabilities, which shows as a growing gap between the loss and what the
  logits imply. Mitigated by a z-loss regularizer on the logsumexp.

The paper's broader finding is the practically useful one: these instabilities
appear in small models at high learning rate, the same mitigations work at
both scales, and **the sensitivity of the loss to learning rate is a usable
predictor** — a model whose loss-vs-lr curve is sharply peaked is a model that
will be unstable when scaled.

GLM-130B (arXiv:2210.02414) documents a real large-scale run's stability
problems and mitigations, which is a useful case study for what actually gets
tried in practice.

## Gradient norm is the better early warning

Loss is a lagging indicator; the gradient norm moves first.

| Gradient-norm behaviour | Reading |
|---|---|
| Stable, slowly decreasing | healthy |
| Occasional spikes, clipped, loss unaffected | normal — clipping is doing its job |
| Spikes growing in frequency | approaching instability; consider lowering lr or extending warmup |
| Sudden order-of-magnitude jump | a bad batch, or a numerical problem |
| Norm → 0 with loss flat | vanishing updates — check bf16 master weights, or a detached graph |
| Norm differs by orders of magnitude across ranks | data skew or a corrupt shard on one rank — **not** a communication problem |

The last row is worth calling out: per-rank gradient-norm divergence is a
*data* signal, and people frequently misroute it to the networking stack. Log
per-rank gradient norm, not just the global one.

## What to log, at minimum

For a distributed run, per step:

```
step, tokens_seen, loss (token-weighted), grad_norm (global),
learning_rate, step_time, tokens/s, MFU,
per-rank: step_time, grad_norm, tokens_processed
occasionally: loss_scale (fp16), memory allocated/reserved
```

The per-rank rows are what make a straggler, a bad shard, and data skew
visible. Without them, all three present identically as "training got slow and
the loss looks odd".

## Sources

- Small-scale proxies for large-scale Transformer training instabilities — arXiv:2309.14322
- GLM-130B: An Open Bilingual Pre-trained Model — arXiv:2210.02414
- Symbolic Discovery of Optimization Algorithms (Lion) — arXiv:2302.06675
