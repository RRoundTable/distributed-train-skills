# Memory Taxonomy and Math

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`. This file **derives**
the activation-memory formula; `parallelism-strategies/references/tensor-parallel.md`
and `sequence-context-parallel.md` cite it.

## The full equation

```
HBM = model_states + activations + fragmentation + framework_overhead
```

### Framework overhead — budget it, do not discover it

| Item | Typical | Notes |
|---|---|---|
| CUDA context | 300–600 MB | per process, before any tensor |
| NCCL communicator buffers | 100 MB – 1 GB+ | grows with world size and the number of communicators |
| cuBLAS / cuDNN workspace | 100 MB – 1 GB | depends on the algorithms selected |
| Kernel/library code | 100–300 MB | |
| Allocator overhead (reserved − allocated) | 0–20% of reserved | fragmentation lives here |

**Budget 2–4 GB before your model allocates anything**, more at large world
size with several process groups. A plan that fills 79 of 80 GB on paper will
OOM.

## Model states

```
model_states = (2 + 2 + K)·Ψ bytes        K = 12 for fp32-master Adam
             = 16Ψ                        → 16Ψ/d under ZeRO-3
```

Derived in `parallelism-strategies/references/data-parallel-and-zero.md`.
`K` by optimizer: `references/optimizer-and-precision-memory.md`.

## Activations, derived

Activations are tensors produced in the forward pass and kept because backward
needs them. Count them per transformer layer, per micro-batch, in bf16
(2 bytes) unless noted. Following arXiv:2205.05198, express everything in
units of `s·b·h` bytes.

### Attention block

| Saved tensor | Size in units of `s·b·h` bytes |
|---|---|
| block input (for the residual and for QKV backward) | 2 |
| `Q`, `K`, `V` (after projection) | 6 |
| `QKᵀ` scores, pre-softmax | `2as/h` |
| softmax output | `2as/h` |
| softmax dropout mask (1 byte per element) | `as/h` |
| attention over `V` (dropout input) | `2as/h` |
| output-projection input | 2 |
| attention dropout mask (1 byte) | 1 |
| **subtotal** | **11 + `5as/h`** |

### MLP block

| Saved tensor | Units |
|---|---|
| block input | 2 |
| first-linear output (`4h` wide, so 8 units) | 8 |
| activation-function output (`4h` wide) | 8 |
| dropout mask (1 byte, `h` wide) | 1 |
| **subtotal** | **19** |

### Layer norms

Two layernorms per layer, each saving its input: **4** units.

### Total

```
per layer = s·b·h·(11 + 5as/h + 19 + 4) = s·b·h·(34 + 5as/h)   bytes
```

```
activations = L · s·b·h·(34 + 5as/h)      bytes per rank, per micro-batch
```

Note the structure: `34` is linear in `s`, while `5as/h` makes the second term
**quadratic** in `s`. The crossover is at `5as/h = 34`, i.e.
`s = 6.8·h/a = 6.8·d_h`. For `d_h = 128` that is `s ≈ 870` — so for essentially
every real training configuration the attention term already dominates.

### With parallelism (same source)

```
TP only:                       s·b·h·(10 + 24/t + 5as/(h·t))
TP + sequence parallelism:     (s·b·h/t)·(34 + 5as/h)
TP + SP + selective recompute: 34·s·b·h/t
```

The `10` in the TP-only row is the layernorm and dropout regions that TP
leaves replicated — see
`parallelism-strategies/references/sequence-context-parallel.md`.

## Worked budget 1 — 7B on one 80 GB H100

```
Ψ = 7e9, L = 32, h = 4096, a = 32, s = 4096, b = 1, bf16
```

**Model states (DDP, `d = 1`):** `16 · 7e9 = 112 GB`. Over budget before
anything else. This is the answer: **7B does not train on one 80 GB GPU with
fp32-master Adam**, regardless of batch size.

**With ZeRO-3 at `d = 8`:** `112/8 = 14 GB`.

**Activations, per layer:**
```
5as/h = 5·32·4096/4096 = 160
per layer = 4096·1·4096·(34+160) = 3.255e9 bytes = 3.25 GB
32 layers = 104 GB                              ← still impossible
```

**With selective recomputation** (drops the `5as/h` term):
```
per layer = 4096·1·4096·34 = 5.70e8 bytes = 0.57 GB
32 layers = 18.2 GB
```

**Total:** `14 (states) + 18.2 (activations) + 3 (overhead) = 35.2 GB`.
Fits, with room to raise `b` to 2 or 3.

## Worked budget 2 — the "13B on 80 GB" question

```
Ψ = 13e9, bf16 + fp32-master Adam
model_states = 16 · 13e9 = 208 GB
```

208 GB does not fit in 80 GB. Lowering `b` does nothing — model states are
independent of batch size. The routes that do work:

| Route | Per-rank states | Viable? |
|---|---|---|
| ZeRO-3, `d = 4` | 52 GB | yes, leaves ~25 GB for activations + overhead |
| ZeRO-3, `d = 8` | 26 GB | comfortable |
| ZeRO-2, `d = 8` | `2Ψ + 14Ψ/8 = 26 + 22.75 = 48.75` GB | yes, and cheaper on the wire than ZeRO-3 |
| 8-bit Adam + bf16 master, single GPU | `2Ψ+2Ψ+2Ψ+2Ψ ≈ 8Ψ = 104` GB | still no |
| LoRA / other PEFT | base weights `2Ψ = 26` GB + tiny adapter state | **yes on one GPU** — but it is not full fine-tuning |
| CPU offload of optimizer, single GPU | `2Ψ + 2Ψ = 52` GB on GPU | yes, at PCIe speed |

The honest answer to "can I train 13B on one 80 GB card" is: **not full
fine-tuning with Adam.** With offload or PEFT, yes. This distinction is worth
making explicitly, because "it fits" and "it fits the thing you meant" differ.

## Worked budget 3 — 70B at `s = 8192`

```
Ψ = 70e9, L = 80, h = 8192, a = 64, s = 8192, b = 1, bf16
```

**States:** `16 · 70e9 = 1120 GB` → ZeRO-3 at `d = 8`: 140 GB (still too
much); at `d = 64`: 17.5 GB.

**Activations, per layer, no model parallelism:**
```
5as/h = 5·64·8192/8192 = 320
per layer = 8192·1·8192·354 = 2.376e10 = 23.8 GB    per layer
```

One layer's activations exceed a whole H100. `t = 8` with SP divides by 8
(2.97 GB/layer, 238 GB total — still impossible); adding selective
recomputation gives `8192·8192·34/8 = 285 MB` per layer, **22.8 GB** for all
80 layers.

**Total at `(d,t,p,c) = (8,8,1,1)`:** `17.5 + 22.8 + 3 = 43.3 GB`. Fits.
This is the arithmetic behind worked configuration 1 in
`parallelism-strategies/references/composing-nd-parallelism.md`.

## What scales with what — the summary table

| Term | `Ψ` | `s` | `b` | `L` | `d` | `t` | `p` | `c` |
|---|---|---|---|---|---|---|---|---|
| weights, grads, optimizer | ∝ | — | — | — | ÷ (ZeRO) | ÷ | ÷ | — |
| activations (the `34` part) | — | ∝ | ∝ | ∝ | — | ÷ (with SP) | ÷ | ÷ |
| activations (the `5as/h` part) | — | ∝ s² | ∝ | ∝ | — | ÷ | ÷ | ÷ |
| fragmentation | — | pattern | pattern | — | — | — | — | — |
| framework overhead | — | — | — | — | grows slightly with `n` | | | |

Read the second row against the first: **`d` (ZeRO) does nothing for
activations.** A job can have almost no parameter memory resident and still
OOM in the middle of a forward pass. That is the single most common surprise
in ZeRO-3 runs.

## Estimation checklist

1. `16Ψ` (or `16Ψ/d`) for states — with the correct `K` for the optimizer.
2. `L · s·b·h·(34 + 5as/h)` for activations, with the parallelism divisor.
3. Subtract the recomputation you are actually using.
4. Add 2–4 GB of framework overhead.
5. Leave 10% headroom for fragmentation.
6. Compare against the card, not against the datasheet capacity minus zero.

## Sources

- Reducing Activation Recomputation in Large Transformer Models — arXiv:2205.05198
- ZeRO — arXiv:1910.02054
