---
name: training-metrics
description: |
  The numbers you measure from a training run and what they mean. Activate for:
  MFU and HFU — how to compute them, what is achievable, why they are inflated
  or deflated; FLOP counting and the 6N-per-token rule; tokens/s and
  samples/s; scaling efficiency, "GPU 8장 썼는데 5.2배밖에 안 빨라졌어",
  strong vs weak scaling; step-time modelling and predicting throughput at
  larger scale; loss curve diagnostics — spikes, plateaus, divergence, "loss가
  갑자기 튀었어", why loss changed after a parallelism config change; gradient
  norm behaviour; compute budgets, Chinchilla and scaling laws, "7B 모델
  토큰 몇 개 학습해야 해"; and how to benchmark two configurations honestly.
  Do NOT activate for: choosing parallelism degrees or ZeRO stage (use
  distributed-train:parallelism-strategies); collectives, NCCL, bucketing or
  hangs (use distributed-train:communication-backends); single-GPU kernels,
  roofline, precision or peak FLOPs of a chip (use distributed-train:gpu-architecture);
  OOM, activation checkpointing or offload (use distributed-train:memory-offloading).
  Also do NOT activate for real cluster operations — reading a running job's
  metrics dashboard, job logs, job status, quota, or submitting a benchmark
  job. That is mlops:forge-train.
---

# Training Metrics

Turning wall-clock, loss, and throughput into a defensible statement about
what is happening. This skill is the measurement half of the plugin; the other
five explain mechanisms, and this one decides which mechanism the evidence
actually supports.

> Platform recipe: retrieving metrics from a running job, its dashboard, or
> its logs is skill `mlops:forge-train`. This skill defines what to compute
> from numbers you already have.

Symbols (`N`, `Ψ`, `D`, `n`, `L`, `h`, `s`, `b`, `m`, `B`, `V`, `T`,
`P_peak`) and the batch invariant `B = b·m·d` are defined in
skill `distributed-train:distributed-training-router`, file
`distributed-training-router/references/notation-and-glossary.md`.

## `6N` FLOPs per token

The single most useful approximation in the field. For a dense transformer
with `N` non-embedding parameters, one token of training costs:

```
forward   ≈ 2N     (one multiply + one add per parameter per token)
backward  ≈ 4N     (gradients w.r.t. inputs and w.r.t. weights ≈ 2× forward)
total     ≈ 6N     FLOPs per token
```

So a full training run of `D` tokens costs `≈ 6ND` FLOPs. For Llama-3-8B on
15T tokens: `6 · 8e9 · 15e12 = 7.2e26` FLOPs.

**Where it breaks.** `6N` counts only the parameterized matmuls. Attention's
score computation has no parameters and scales as `s²`. Per layer:

```
parameterized  ≈ 12·h²   FLOPs per token
attention      ≈ 2·s·h   FLOPs per token  (QKᵀ and PV together, per token)
ratio          = 2sh / 12h² = s/(6h)
```

Attention exceeds 50% of total FLOPs when `s > 6h`. For `h = 4096`, that is
`s > 24576`. Below that, `6N` is a good approximation; well above it, `6N`
undercounts substantially and any MFU computed from it is wrong. Megatron's
own per-iteration formula carries this correction explicitly:

```
F = 96 · B · s · L · h² · (1 + s/(6h) + V/(16·L·h))       (arXiv:2104.04473)
```

The `96` includes a full extra forward pass for activation recomputation;
without recomputation the coefficient is `72`, which is exactly `6N` with
`N ≈ 12Lh²`. `96/72 = 4/3 = 8/6` — the same ratio as `HFU/MFU` under full
recompute. Derivation and the correction terms:
`references/flop-counting-and-mfu.md`.

## MFU

```
MFU = (6 · N · D_step) / (T · P_peak · n)
```

where `D_step = B · s` tokens per step, `T` is seconds per step, `P_peak` is
one accelerator's **dense** peak FLOP/s at the training dtype, and `n` is the
world size.

**MFU vs HFU.** HFU counts every FLOP the hardware executed; MFU counts only
the FLOPs the model definition requires. They differ whenever work is repeated:

```
under full activation recomputation:   HFU/MFU ≈ 8/6 ≈ 1.33
```

because recomputation adds an extra forward (`2N`) to the `6N`. MFU ≤ HFU
always. MFU is the number to compare across systems, because it is invariant
to your memory-saving choices; HFU tells you how efficiently the chip ran the
work you gave it.

**The three ways MFU is inflated** — check all three before believing a
number, yours or anyone's:

1. **Sparse peak.** NVIDIA quotes H100 at 1,979 TFLOPS bf16 *with sparsity*.
   Dense is 989. Using the sparse figure halves your MFU; using it in the
   denominator by mistake and reporting the result halves the number you
   should have reported. Dense transformer training gets the dense peak.
2. **Counting recomputation.** Including the recompute forward in the
   numerator makes MFU into HFU, a free 33% under full recompute.
3. **Including embeddings in `N`.** `6N` is calibrated on non-embedding
   parameters. At small `h` and large `V` the embedding is a large fraction of
   the parameter count and inflates the numerator.

Achievable, from published runs (all verified):

| Run | Hardware | MFU |
|---|---|---|
| PaLM 540B | 6144 TPU v4 | **46.2%** (57.8% HFU) |
| Megatron PTD-P 1T | 3072 A100 | 52% of peak, 163 TFLOP/s/GPU |
| MegaScale 175B | 12288 GPUs | **55.2%** |
| Llama 3 405B | up to 16384 H100 | **38–43%** BF16 |

A claim above ~60% on a dense transformer deserves an audit of the three
inflations. A number below 25% at scale usually means communication or
bubbles, not kernels.

## The step-time model

The model that turns three measurements into a prediction:

```
T = T_compute/n_eff  +  T_comm(n)  +  T_bubble(p)  +  T_straggler
```

Fit it from measurements at three world sizes and you can *predict* larger
scale rather than extrapolate a line. The fitting procedure, and how each term
maps to a sibling skill, is in
`references/throughput-and-scaling-efficiency.md`. Its main value is
falsifiability: a prediction that misses tells you which term you got wrong.

## Scaling efficiency

```
E(n) = T(n_ref) · n_ref / (T(n) · n)          strong scaling
E(n) = T(n_ref) / T(n)                        weak scaling (work grows with n)
```

Always say which one you measured. "8 GPUs gave 5.2×" is 65% strong-scaling
efficiency — which may be fine or terrible depending on whether the 8 GPUs
span a node boundary and on the compute-to-communication ratio
`6·b·m·s / bytes_per_elem`. Diagnosis order is in the reference file.

## Loss diagnostics

Two categories, and conflating them wastes days:

**Optimization problems** — spikes, plateaus, divergence. Distributed-specific
causes include a bad shard, per-rank RNG collision, a straggler dropping
samples, and a stale checkpoint resumed with a different data order. The
taxonomy is in `references/loss-curve-diagnostics.md`, along with the
small-scale-proxy findings (arXiv:2309.14322) on logit growth and
attention-entropy collapse.

**Correctness problems** — the loss changed and it should not have. The
checklist when loss moves after a parallelism change:

1. Is `B = b·m·d` **identical**? Changing `t`, `p`, or `c` must not change
   `B`; if it did, this is a different experiment.
2. Is the loss a mean over **tokens** or a mean of per-rank means? With
   unequal token counts per rank (padding, variable length), mean-of-means is
   biased. Reduce `(sum_loss, count)` and divide once.
3. Is per-rank RNG seeded so that dropout differs across DP ranks but is
   **identical** across TP/PP/CP ranks (which must see the same mask)?
4. Is the dataloader sharded so no sample is seen twice per epoch, and does
   the shuffle depend on `d`?
5. Are gradients normalized by the same total token count on both sides?

Any one of these produces a plausible-looking but different curve, which is
why "the loss moved a little" after a config change is never acceptable.

## Compute budget

Chinchilla (arXiv:2203.15556): at a fixed compute budget `C ≈ 6ND`, loss is
minimized with `N` and `D` scaled roughly equally — about **20 tokens per
parameter**. Modern practice deliberately overtrains far past that (Llama 3
trained 8B on ~15T tokens, ~1875 tokens/parameter) because inference cost
favours a smaller model, and the training-optimal point is not the
deployment-optimal point. Details and the caveats:
`references/compute-budget-and-scaling-laws.md`.

## Rules for answering here

- Compute MFU explicitly, showing `N`, `D_step`, `T`, `P_peak` (dense), and
  `n`. An MFU without its inputs is not checkable.
- Say strong or weak scaling every time you report an efficiency.
- Distinguish MFU from HFU whenever recomputation is in play.
- Before diagnosing a throughput problem, confirm the measurement is real:
  warm-up excluded, several steps, a stated percentile.
- When loss moves after a config change, run the correctness checklist before
  the optimization taxonomy.
- Do not emit `forge` commands or read a live job's state.

## Reference files

| File | Contents |
|---|---|
| `references/flop-counting-and-mfu.md` | `6N` derived, correction terms, MFU/HFU, the three inflations, worked examples, MoE and `N_active` |
| `references/throughput-and-scaling-efficiency.md` | strong vs weak scaling, the step-time model fitted from three points, diagnosis order, worked prediction |
| `references/loss-curve-diagnostics.md` | spike taxonomy, distributed-specific causes, the correctness checklist, gradient-norm reading, instability literature |
| `references/compute-budget-and-scaling-laws.md` | Chinchilla, overtraining, budget arithmetic, what scaling laws do and do not predict |
| `references/benchmarking-methodology.md` | the minimum honest protocol, warm-up, variance, what to hold fixed, how to compare two configs |
