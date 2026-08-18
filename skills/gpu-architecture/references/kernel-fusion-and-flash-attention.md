# Kernel Fusion and FlashAttention

Symbols: router → `distributed-training-router/references/notation-and-glossary.md`. This file is where
**online softmax is derived**; `parallelism-strategies/references/sequence-context-parallel.md`
cites it for ring attention.

## The economics of fusion

Two consecutive memory-bound elementwise kernels each read their input from
HBM and write their output back:

```
unfused:   read x, write y        then   read y, write z     = 4 HBM passes
fused:     read x, compute both, write z                     = 2 HBM passes
```

Since both kernels are memory-bound, time is proportional to HBM traffic, so
fusion gives a ~2× speedup on that sequence — despite doing identical
arithmetic. Chain `k` elementwise ops and fusion gives roughly `k×`.

Fusion also removes `k−1` kernel launches (a few microseconds each — material
when a step has thousands) and keeps intermediates in registers or shared
memory rather than HBM.

What fusion **cannot** do: help a compute-bound kernel. Fusing two large GEMMs
saves the intermediate write but the GEMMs still dominate. This is why fusion
targets the memory-bound rows of the AI table in
`references/roofline-and-arithmetic-intensity.md` — norms, activations,
residuals, softmax, optimizer steps — and not the four big matmuls.

Standard fusion targets in a transformer:

| Fusion | Passes saved |
|---|---|
| bias + GeLU/SiLU | 2 |
| LayerNorm/RMSNorm (mean, var, normalize, affine) | ~3 |
| residual add + norm | 2 |
| dropout + residual + norm | ~4 |
| attention (the big one — see below) | many |
| multi-tensor / fused Adam | ~2 per state tensor |
| cross-entropy + logsoftmax (avoids materializing logits gradients) | large at big `V` |

## FlashAttention: the problem

Naive attention:

```
S = Q Kᵀ            materialize  [b, a, s, s]   → HBM
P = softmax(S)      read S, write P             → HBM
O = P V             read P                      → HBM
```

FLOPs are `~4·b·a·s²·d_h`. Bytes are `~O(b·a·s²)`. So
`AI ≈ d_h ≈ 64–128` — well below H100's ridge point of 295. Attention is
memory-bound, and the `s²` intermediate is also what makes long context OOM.

At `b=1`, `a=32`, `s=8192`, bf16, that intermediate is **4.3 GB per layer**,
written and read several times.

## Online softmax, derived

The obstacle to tiling is that softmax needs a global maximum and a global sum
over the whole row before it can normalize. Online softmax removes that
dependency by maintaining running statistics and rescaling.

Numerically stable softmax over `x₁ … x_N`:

```
M = max_i x_i
Z = Σ_i exp(x_i − M)
softmax(x)_i = exp(x_i − M) / Z
```

Now process the row in blocks. After block `j`, keep the running max `M_j`,
the running sum `Z_j`, and the running weighted output `O_j`. When block
`j+1` arrives with local max `m` and local sum `z = Σ exp(x − m)`:

```
M_new = max(M_j, m)

Z_new = Z_j · exp(M_j − M_new)  +  z · exp(m − M_new)

O_new = O_j · exp(M_j − M_new)  +  (Σ_block exp(x − M_new)·v)
```

Every term carries a correction factor `exp(M_old − M_new)` that retroactively
rebases the accumulated values onto the new maximum. Because `exp` turns
addition into multiplication, the rebasing is a single multiply — the
accumulator never needs the original scores again.

At the end, divide `O` by `Z`. The result is **exactly** the same as computing
softmax over the full row: this is an algebraic identity, not an
approximation.

The consequence: the `s × s` score matrix never has to exist. Blocks of `S`
are computed in shared memory, consumed immediately, and discarded.

## FlashAttention: the algorithm

```
partition Q into blocks Q_i  (rows), K and V into blocks K_j, V_j (columns)
for each Q_i:
    load Q_i into shared memory; initialize O_i = 0, M_i = -inf, Z_i = 0
    for each K_j, V_j:
        load K_j, V_j into shared memory
        S_ij = Q_i K_jᵀ                      # in SMEM/registers, never HBM
        update M_i, Z_i, O_i by online softmax with S_ij and V_j
    write O_i / Z_i to HBM
```

HBM traffic drops from `O(s²)` to `O(s·d_h)` per head plus the `Q`/`K`/`V`/`O`
tensors themselves. Arithmetic intensity rises by roughly the block factor,
moving attention across the ridge point. FLOPs are unchanged — this is a pure
memory-traffic win, which is why it is called an *IO-aware* algorithm.

For the backward pass, the score matrix would normally be needed. FlashAttention
**recomputes** it blockwise from `Q`, `K`, `V` and the saved per-row statistics
`(M, Z)` — trading a small amount of extra math for not storing `s²` values.
That is the same recompute-for-memory trade as activation checkpointing
(`memory-offloading/references/activation-recomputation.md`), applied inside a
single kernel.

Reported speedups and the memory result: FlashAttention reduces attention's
memory from quadratic to linear in `s`, which is what made long-context
training practical at all (arXiv:2205.14135).

## Versions

| Version | Main change | Source |
|---|---|---|
| FlashAttention | tiling + online softmax + backward recompute | arXiv:2205.14135 |
| FlashAttention-2 | better work partitioning across warps and thread blocks; fewer non-matmul FLOPs; parallelize over sequence length as well as batch/heads | arXiv:2307.08691 |
| FlashAttention-3 | Hopper-specific: asynchrony via TMA and `wgmma`, warp specialization (producer/consumer), fp8 support | arXiv:2407.08608 |

The progression is instructive: v1 fixed the *algorithm* (HBM traffic), v2
fixed the *parallelization* (SM utilization, especially at small batch), and
v3 exploits *hardware asynchrony*. Each addressed a different roofline
limiter on the same operation.

## Causal masking and block skipping

With a causal mask, blocks entirely above the diagonal contribute nothing and
can be **skipped**, roughly halving the work. This means the causal cost is
not `s²` but `~s²/2` — and it is also the source of the load imbalance in
context parallelism, where contiguous sharding gives the last rank far more
surviving blocks than the first
(`parallelism-strategies/references/sequence-context-parallel.md`).

## torch.compile and Triton

Hand-writing fused kernels does not scale to every model. `torch.compile`
captures a graph (TorchDynamo, via Python bytecode transformation) and lowers
it through TorchInductor, which generates Triton kernels for GPU and C++ for
CPU. The reported result is a 2.27× inference / 1.41× training geomean
speedup on an A100 across 180+ real models
(*PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode
Transformation and Graph Compilation*, ASPLOS 2024, doi 10.1145/3620665.3640366).

What it does well: fusing the memory-bound elementwise chains — exactly the
rows of the AI table that are starved. What it does not do: beat cuBLAS on
large GEMMs, or invent FlashAttention (it dispatches to an existing fused
attention kernel).

Practical notes that matter for training runs:

- **Graph breaks** kill the benefit. Data-dependent control flow, `.item()`,
  printing a tensor, or an unsupported op splits the graph and each fragment
  is compiled separately with HBM round-trips at the boundaries.
- **Dynamic shapes** trigger recompilation. Variable sequence length is the
  usual culprit; bucketing lengths bounds the number of compilations.
- **Compile time is real** — the first step (or first few) is much slower.
  Exclude it from every benchmark, or you will measure the compiler.

## When fusion is the answer, and when it is not

| Situation | Fusion helps? |
|---|---|
| Many small elementwise ops between GEMMs | **yes** — the primary case |
| Attention dominating the profile | **yes** — use a fused attention kernel |
| One large GEMM dominating | no — it is compute-bound already |
| Communication dominating | no — `distributed-train:communication-backends` |
| OOM | indirectly — fusion removes intermediates, which does free memory |
| Launch overhead / thousands of tiny kernels | **yes** — fewer launches |
| Optimizer step is a visible fraction of the step | **yes** — fused/multi-tensor optimizer |

## Sources

- FlashAttention — arXiv:2205.14135
- FlashAttention-2 — arXiv:2307.08691
- FlashAttention-3 — arXiv:2407.08608
- PyTorch 2 / TorchInductor — ASPLOS 2024, doi 10.1145/3620665.3640366
