# Compute Budget and Scaling Laws

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`. `C` is total training
compute in FLOPs, `N` parameters, `D` tokens.

## The budget identity

```
C ≈ 6 · N · D          FLOPs for a run of D tokens on an N-parameter model
```

Everything below is a statement about how to spend a fixed `C` between `N` and
`D`. Derivation of `6N`: `references/flop-counting-and-mfu.md`.

## Chinchilla

Hoffmann et al. (arXiv:2203.15556) fit loss as a function of `N` and `D` and
minimize under the constraint `C = 6ND`. The result: at the compute-optimal
point, `N` and `D` should scale **roughly equally** with `C` — approximately

```
D ≈ 20 · N            tokens per parameter, at the training-compute optimum
```

The empirical demonstration was Chinchilla (70B, 1.4T tokens) outperforming
Gopher (280B, 300B tokens) at comparable compute — a 4× smaller model trained
on 4.7× more data.

Consequences at the time: most models before it were badly undertrained, and
the field's default parameter-to-token ratio shifted by an order of magnitude.

## Why nobody follows it anymore

Chinchilla minimizes **training** compute for a target loss. It says nothing
about inference. If a model will serve many tokens after training, the total
cost is:

```
total = C_train + C_inference = 6ND_train + 2N·D_served
```

Inference cost is linear in `N` and does not depend on `D_train` at all. So
for a heavily-served model it is rational to train a *smaller* model for
*much* longer than Chinchilla-optimal, accepting worse loss-per-training-FLOP
in exchange for permanently cheaper inference.

| Model | `N` | `D` | tokens/param | vs Chinchilla |
|---|---|---|---|---|
| Gopher 280B | 280e9 | 300e9 | ~1 | badly undertrained |
| Chinchilla 70B | 70e9 | 1.4e12 | 20 | the optimum |
| Llama 3 8B | 8e9 | ~15e12 | ~1875 | ~94× overtrained, deliberately |
| Llama 3 405B | 405e9 | ~15e12 | ~37 | near-optimal-ish |

Llama 3 8B is the clearest statement of the modern position: ~94× past the
training-optimal token count, because the deployment economics dominate
(arXiv:2407.21783).

**So the right question is never "what does Chinchilla say".** It is: what is
the total cost over the model's life, and what is the largest model the
serving budget tolerates?

## Budget arithmetic

Given a cluster and a deadline, what can you train?

```
available FLOPs = P_peak · n · MFU · seconds
tokens          = available FLOPs / (6N)
```

**Worked.** 512× H100 for 30 days at 40% MFU, training a 70B model:

```
seconds  = 30 · 86400 = 2.592e6
FLOPs    = 989e12 · 512 · 0.40 · 2.592e6 = 5.25e23
tokens   = 5.25e23 / (6 · 70e9) = 1.25e12 = 1.25 T tokens
```

At `D/N = 1.25e12/70e9 = 18`, that is right about the Chinchilla point — so
this budget can train a compute-optimal 70B and no more. If you want an
overtrained 8B instead:

```
tokens = 5.25e23 / (6 · 8e9) = 1.09e13 = 10.9 T tokens   (1370 tokens/param)
```

Same cluster, same month, a very different model. The choice is a product
decision informed by this arithmetic, not a research one.

**Inverting for the MFU you need.** If the deadline is fixed and `N`, `D` are
fixed:

```
required MFU = 6ND / (P_peak · n · seconds)
```

This is often the most useful form: it converts "we need to finish by X" into
a concrete efficiency target that the other five skills can be aimed at. If
the required MFU exceeds ~50%, the plan needs more GPUs or fewer tokens, not
better engineering.

## What scaling laws do and do not predict

**They do predict**, within the fitted regime:

- pretraining loss as a smooth function of `N`, `D`, and `C`
- the compute-optimal allocation between `N` and `D`
- that loss improvements are *predictable* — a useful sanity check on a run in
  progress, and the basis for deciding a run is underperforming before it
  finishes

**They do not predict:**

- downstream task performance — loss and benchmark scores decouple, especially
  after instruction tuning
- performance outside the fitted range (extrapolation many orders of magnitude
  past the fit is an assumption, not a result)
- anything about data quality, which is not a variable in the law; a law fitted
  on one corpus does not transfer to another
- MoE models directly — `N_active` and `N_total` enter differently, and the
  dense laws do not apply unmodified
  (`parallelism-strategies/references/expert-parallel-moe.md`)
- optimizer or architecture changes — the constants are refitted, not carried
  over
- the effect of learning-rate schedule, warmup, or batch size, all of which are
  held at "reasonable" values in the fits

## Batch size and the critical batch

Throughput is not progress. Beyond a **critical batch size**, doubling `B`
stops halving the number of steps to a target loss, so the extra samples buy
little. Practical markers:

- The critical batch grows as training proceeds (the gradient noise scale
  increases as the loss falls), which is why some large runs ramp `B` upward
  over the run rather than fixing it.
- Learning rate must scale with `B` — linear scaling with warmup for SGD-like
  regimes, square-root-ish for Adam-like — or a larger batch is simply a
  smaller effective learning rate.
- A parallelism change that alters `B` (see
  `references/loss-curve-diagnostics.md`) therefore also alters the effective
  learning rate. This is why `B` must be held fixed when comparing configs.

The consequence for this plugin's other skills: raising `m` to improve the
compute-to-communication ratio is free only while `B` stays under the critical
batch. Past it, you are buying throughput with sample efficiency.

## Sources

- Training Compute-Optimal Large Language Models (Chinchilla) — arXiv:2203.15556
- PaLM — arXiv:2204.02311
- The Llama 3 Herd of Models — arXiv:2407.21783
- DeepSeek-V3 Technical Report — arXiv:2412.19437
