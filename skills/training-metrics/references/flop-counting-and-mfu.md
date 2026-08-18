# FLOP Counting and MFU

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`. This file **derives**
`6N` and `MFU`; other files cite it.

## `6N` per token, derived

Consider a single weight matrix `W ∈ R^{h_in × h_out}` and one token's
activation `x ∈ R^{h_in}`.

**Forward.** `y = xW` requires `h_in × h_out` multiplies and the same number
of adds:

```
forward FLOPs = 2 · (number of parameters in W)
```

**Backward.** Two gradients are needed:

```
∂L/∂x = (∂L/∂y) Wᵀ        →  2 · |W| FLOPs
∂L/∂W = xᵀ (∂L/∂y)        →  2 · |W| FLOPs
backward total            =  4 · |W|
```

Summing, and summing over every weight matrix in the model:

```
FLOPs per token = 2N + 4N = 6N
FLOPs for a run of D tokens = 6ND
```

`N` here is the count of parameters that participate in a matmul against the
token — the **non-embedding** parameters. Embedding lookups are gathers, not
matmuls, and cost ~0 FLOPs (the output projection *is* a matmul and does
count).

For a standard transformer, `N ≈ 12·L·h²` (attention QKVO gives `4h²`, the MLP
with a 4× expansion gives `8h²`).

## The corrections

`6N` omits everything without parameters. The two that matter:

**Attention scores.** Per layer, per token, `QKᵀ` and `PV` together cost
`≈ 2·s·h` FLOPs in the forward (`4h·s` counting both, `2sh` per the standard
accounting used in the Megatron formula). Relative to the layer's `12h²`
parameterized FLOPs:

```
attention fraction = 2sh / (12h²) = s/(6h)
```

```
attention exceeds 50% of FLOPs when  s > 6h
```

| `h` | `s` at which attention = 50% of FLOPs |
|---|---|
| 2048 (1.3B class) | 12288 |
| 4096 (7B class) | 24576 |
| 8192 (70B class) | 49152 |
| 16384 (405B class) | 98304 |

Below the threshold, `6N` is a good approximation. Well above it — long-context
training — `6N` undercounts and every MFU derived from it is understated.

**Output projection / vocabulary.** The final `h → V` projection contributes
`V/(16·L·h)` relative to the rest in Megatron's formulation. At `V = 128256`,
`L = 32`, `h = 4096`, that is `128256/(16·32·4096) = 6.1%` — not negligible for
small models with large vocabularies.

## Megatron's per-iteration formula

```
F = 96 · B · s · L · h² · (1 + s/(6h) + V/(16·L·h))
```

(arXiv:2104.04473.) Reading the coefficient:

- `72 = 6 × 12` corresponds to `6N` with `N = 12Lh²` — forward plus backward,
  **no recomputation**.
- `96 = 8 × 12` corresponds to `8N` — forward, backward, **plus a second
  forward** for full activation recomputation.

So `96` is a **hardware** FLOP count (HFU-style) and `72` is a **model** FLOP
count (MFU-style). Their ratio is `96/72 = 4/3 ≈ 1.33`, which is exactly the
`HFU/MFU` ratio under full recompute. When comparing published numbers, check
which coefficient was used — it is a 33% difference.

## MFU and HFU

```
MFU = model FLOPs    / (T · P_peak · n)
HFU = hardware FLOPs / (T · P_peak · n)
```

with `model FLOPs per step = 6 · N · B · s` (plus corrections), `T` seconds
per step, `P_peak` the **dense** peak of one accelerator at the training
dtype, and `n` the world size.

```
MFU ≤ HFU  always.   Equality iff no work is repeated.
```

| Situation | `HFU/MFU` |
|---|---|
| no recomputation | 1.00 |
| selective recomputation (attention only) | ~1.02–1.05 |
| full recomputation | ~1.33 |

MFU is the cross-system comparison metric: it is invariant to whether you
chose to recompute. HFU measures how well the chip ran whatever you asked of
it. Report both when recomputation is on; report which one when reporting one.

## The three inflations

**1. Sparse peak in the denominator.** NVIDIA's headline H100 figure is 1,979
TFLOPS bf16 **with sparsity**; dense is 989. Dense transformer training gets
the dense number. Using 1,979 halves your reported MFU (making a good run look
bad); a published number computed against 989 is not comparable to one
computed against 1,979. Always state the peak you used.

**2. Recompute in the numerator.** Counting the recompute forward turns MFU
into HFU — a free 33% under full recompute. If a number is suspiciously high
and recomputation is on, this is the first thing to check.

**3. Embeddings in `N`.** `6N` is calibrated on non-embedding parameters. For
a 1B model with `V = 128256` and `h = 2048`, the embedding is `2 × 128256 ×
2048 = 525M` parameters — over a third of the total. Including it in `N`
inflates the numerator by that fraction.

There is one common *deflation* too: **ignoring the attention correction at
long context**. At `s = 131072` and `h = 16384`, `s/(6h) = 1.33`, so true
FLOPs are ~2.3× the `6N` estimate and true MFU is ~2.3× what you computed.

## Comparing MFU across *architectures*

MFU is invariant to your memory and scheduling choices. It is **not** invariant
to the model definition, because the model definition is the numerator. Two
systems reporting MFU on "a 175B model" are comparable only if the 175B models
compute the same function.

Two changes that appear in efficiency-oriented papers and both move MFU:

- **Restructured blocks** (e.g. a parallel transformer block, which computes
  attention and the MLP from the same normalized input so they can run
  concurrently). Same parameter count, same `6N`, fewer serialization points —
  MFU rises, and this one is a fair systems gain.
- **Sparse or windowed attention.** `O(s·w)` instead of `O(s²)` changes what
  the model *is*. The numerator must change with it: the `s/(6h)` correction
  no longer applies, and roughly `w/s` of it remains. Keeping the dense
  attention term in the numerator while executing the windowed kernel inflates
  MFU exactly like counting recomputation does.

So when reading a headline MFU alongside an ablation table, check what the
model was at the top of the table and at the bottom. MegaScale's 55.2% at
12,288 GPUs (arXiv:2402.15627) comes with both changes enabled: the parallel
transformer block and sliding-window attention account for 4.6 and 1.0 of its
17.6 points. The systems work is real and separately itemized — but the number
to place next to PaLM's 46.2% or Llama 3's 38–43% is not the same kind of
number, and the paper does not state which attention cost its numerator uses.

The rule that survives: **an MFU is comparable only against another MFU whose
numerator you can reconstruct.** If you cannot reconstruct it, quote it as
"reported" and say what the model was.

## Worked example — 7B on 8× H100

```
N        = 6.7e9 (non-embedding)          h = 4096,  L = 32
b = 4, m = 8, d = 8   →  B = b·m·d = 256   (t = p = c = 1, so n = d = 8)
s        = 4096
D_step   = B · s = 1,048,576 tokens
T        = 18.9 s per step (median of steps 10–30, warm-up excluded)
P_peak   = 989e12 FLOP/s   (H100 SXM, bf16, DENSE — not the 1979 sparse figure)
n        = 8

model FLOPs = 6 · 6.7e9 · 1.048576e6           = 4.217e16
correction  = 1 + s/(6h) = 1 + 4096/24576      = 1.167
model FLOPs (corrected)                        = 4.921e16

denominator = T · P_peak · n = 18.9 · 989e12 · 8 = 1.495e17

MFU = 4.921e16 / 1.495e17 = 32.9%
```

Sanity checks on the result:

- 32.9% is plausible for a single node with no recomputation and no model
  parallelism — below the 38–43% Llama 3 reports at far larger scale, which
  is a reasonable place for an untuned single-node run.
- Had the sparse peak (1979e12) been used, the same run would report **16.5%**
  and look broken. Had recomputation been on and its FLOPs counted, it would
  report **43.8%** and look excellent. Neither number would be wrong
  arithmetic — they would be different metrics.
- The formula is self-checking: an MFU above 100% means an input is wrong,
  almost always `T` (measured with warm-up included, or on the wrong shape) or
  `B` (world size used in place of `d`).

## Worked example — throughput target from an MFU target

Going the other direction is often more useful. What step time should a 70B
model on 64× H100 at 40% MFU achieve, with `B = 512`, `s = 8192`?

```
D_step      = 512 · 8192 = 4.194e6 tokens
model FLOPs = 6 · 70e9 · 4.194e6 = 1.761e18
correction  = 1 + 8192/(6·8192) = 1.167   → 2.055e18
available   = 0.40 · 989e12 · 64 = 2.532e16 FLOP/s
T           = 2.055e18 / 2.532e16 = 81.2 s per step
tokens/s    = 4.194e6 / 81.2 = 51,650 tokens/s
```

Now you have a target to measure against. A measured 120 s means you are at
27% MFU and something in the four budgets is binding.

## MoE: use `N_active`

For a mixture-of-experts model, `6N` must use the parameters **touched per
token**:

```
model FLOPs per token ≈ 6 · N_active
```

Using `N_total` inflates MFU by roughly `E/k`. DeepSeek-V3 has 671B total and
37B activated (arXiv:2412.19437) — an 18× difference. See
`parallelism-strategies/references/expert-parallel-moe.md`.

Note that HFU for MoE is subtler still: capacity-factor padding means the
hardware executes FLOPs on slots that hold no real token, so HFU can exceed
MFU for reasons unrelated to recomputation.

## Checklist for any MFU number

1. Is `N` non-embedding? For MoE, is it `N_active`?
2. Is `P_peak` **dense** and at the actual training dtype?
3. Is `T` from steady-state steps, warm-up excluded, median over several?
4. Is the attention correction applied when `s` is large relative to `h`?
5. Is recomputation on? Then say MFU or HFU explicitly.
6. Is `D_step = B·s` with `B = b·m·d` — and does `d` exclude `t`, `p`, `c`?
7. Is `n` the total accelerator count, not the number of nodes?

Item 6 is the most common error in practice: using the world size in place of
`d` when computing `B` inflates the token count by `t·p·c`.
