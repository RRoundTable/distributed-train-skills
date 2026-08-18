# distributed-train-skills

[![Claude Code plugin](https://img.shields.io/badge/Claude%20Code-plugin-D97757)](https://docs.claude.com/en/docs/claude-code/plugins)
[![version](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FRRoundTable%2Fdistributed-train-skills%2Fmain%2F.claude-plugin%2Fplugin.json&query=%24.version&label=version&color=blue)](.claude-plugin/plugin.json)
[![skills](https://img.shields.io/badge/skills-6-blue)](#the-six-skills)
[![trigger evals](https://img.shields.io/badge/trigger%20evals-135%2F135-brightgreen)](#development)
[![head-to-head](https://img.shields.io/badge/vs%20forge--train-12%2F12-brightgreen)](#development)
[![license](https://img.shields.io/github/license/RRoundTable/distributed-train-skills)](LICENSE)

A Claude Code plugin that ships **distributed-training knowledge** — theory,
math, paper insights, and debugging playbooks — as six skills Claude loads on
demand.

It is deliberately the *complement* to a platform plugin. Nothing here submits
a job, checks quota, builds an image, or reads a cluster log. Every file
defers those to `mlops:forge-train`.

## Install

```bash
claude plugin marketplace add RRoundTable/distributed-train-skills
claude plugin install distributed-train@distributed-train-skills
```

Or from a local clone:

```bash
claude plugin marketplace add /path/to/distributed-train-skills
claude plugin install distributed-train@distributed-train-skills
```

## The six skills

| Skill | Owns |
|---|---|
| `distributed-train:distributed-training-router` | vague / multi-topic / "where do I start" questions; the shared notation; the four-budget diagnostic frame; the symptom index |
| `distributed-train:parallelism-strategies` | sharding vs replication, TP·PP·DP·CP·EP degrees, ZeRO stage, mesh design |
| `distributed-train:communication-backends` | anything on the wire — collective algorithms, α-β cost model, topology, overlap, hangs, stragglers, compression, fault tolerance and checkpoint economics |
| `distributed-train:gpu-architecture` | inside one GPU — roofline, arithmetic intensity, tensor cores, precision, fusion, FlashAttention, profiling |
| `distributed-train:training-metrics` | measured numbers — FLOP counting, MFU/HFU, scaling efficiency, loss curves, scaling laws, benchmarking |
| `distributed-train:memory-offloading` | capacity — the HBM budget, OOM triage, recomputation, offload, fragmentation |

Each skill's `SKILL.md` is the routing and summary layer; the substance lives
in its `references/` files, loaded only when needed.

## What it looks like in use

```
you › NCCL all-reduce가 ring이랑 tree랑 뭐가 달라?
      → communication-backends → references/collective-algorithms.md

you › A100 80GB에 13B bf16 + Adam 학습 가능해?
      → memory-offloading → references/memory-taxonomy-and-math.md

you › Explain why pipeline parallelism has a bubble
      → parallelism-strategies → references/pipeline-parallel.md

you › GPU 8장 썼는데 5.2배밖에 안 빨라졌어
      → training-metrics → references/throughput-and-scaling-efficiency.md

you › What MFU should a 7B model reach on H100?
      → gpu-architecture → references/roofline-and-arithmetic-intensity.md

you › 분산학습 처음인데 뭐부터 봐야 해?
      → distributed-training-router
```

And, when `mlops` is also installed, these go to **forge-train**, not here:

```
GPU 4장짜리 job 띄워줘  ·  job이 QUEUED에서 안 넘어가  ·  forge quota 확인
방금 실패한 job 로그 봐줘  ·  multi-node submit 하는 법
```

## The ownership boundary

The design constraint that governs every file in this repo:

| Surface | Owner |
|---|---|
| Sharding choice, parallelism degrees, ZeRO stage | `parallelism-strategies` |
| Anything between GPUs or nodes | `communication-backends` |
| Inside one GPU: SMs, HBM, peak FLOPs, precision | `gpu-architecture` |
| Measured numbers, curves, profiles | `training-metrics` |
| Memory capacity, OOM, offload, activation recomputation | `memory-offloading` |
| Checkpoint cost and interval, fault recovery, node health checks | `communication-backends` |
| Vague / multi-topic / "where do I start" | `distributed-training-router` |
| **Any real job on a cluster** — submit, quota, images, disks, logs, launcher flags, exit codes | **`mlops:forge-train`** |

Hard rule, enforced across every file: **no `forge` commands, no
cluster-specific env-var values, no launcher recipes.** Where a reader needs
one, the file emits a pointer back to `mlops:forge-train`. Naming
`NCCL_ALGO` or `TORCH_NCCL_TRACE_BUFFER_SIZE` to explain a *mechanism* is in
scope; telling you what to set it to on your cluster is not.

## Shared spine

The six skills compose rather than repeat:

- **One symbol table**, in
  `skills/distributed-training-router/references/notation-and-glossary.md` —
  `Ψ, N, L, h, a, s, b, B, V, d, t, p, c, e, m, v, K, α, β`, the mesh
  invariant `n = d·t·p·c`, the batch invariant `B = b·m·d`,
  bytes-per-element, and the Gb/s-vs-GB/s and `busbw`-vs-`algbw` traps.
  Every other file cites it instead of redefining.
- **The four-budget frame** — compute / capacity / bandwidth / serialization —
  with a three-observation test that separates them. The symptom→skill routing
  table is built on it. Past a few hundred GPUs it extends to
  `goodput = MFU × ETTR`: the four budgets price one step, and availability
  prices how many steps ran at all.
- **Derive once, cite elsewhere.** ZeRO's `2Ψ+2Ψ+KΨ` derives in
  `parallelism-strategies`; the activation formula `s·b·h·(34 + 5as/h)`
  derives in `memory-offloading`; ring all-reduce's `2(n−1)S/n` bound derives
  in `communication-backends`; online softmax derives in `gpu-architecture`;
  `6N` and MFU derive in `training-metrics`.

## Citations

Every arXiv ID and every load-bearing number in this repo was web-verified
before it was written. Anything that failed verification was dropped rather
than hedged — for example, no B200 BF16 ridge point is printed, because
NVIDIA's official DGX B200 specification page does not publish a dense BF16
figure.

## Development

Two eval suites, both defaulting to **`claude-sonnet-5`** — the model most
users hit, and the one where a weak description or a buried fact actually
fails.

**Trigger evals** — does the description fire?

```
/eval-trigger parallelism-strategies            # sonnet by default
/eval-trigger parallelism-strategies claude-opus-5
```

Also carries a **head-to-head** stage that puts all six of our `SKILL.md`
files in one stub alongside a copy of forge-train's and grades on *which*
skill was invoked — the only test that proves cluster-operations queries are
not crowded out.

**Behavioral evals** — once it fires, does it change the answer?

```
/eval-behavior parallelism-strategies
```

Every case runs twice, with and without the skill's content injected, and the
**gap** is the number that matters. 36 cases / ~150 assertions across the six
skills, written against specific checkable quantities (`(p−1)/(m+p−1)`,
`s·b·h·(34 + 5as/h)`, the H100 dense ridge point of 295) rather than vague
ones — a base model already says "communication overhead matters", so an
assertion that accepts that measures nothing.

A case that passes *without* the skill is a signal to tighten the assertion or
cut the content; a case that fails *with* it is a content bug in the reference
files.

Validate the plugin:

```bash
claude plugin validate . --strict
claude plugin tag .              # checks both manifests agree
claude plugin details distributed-train
```

Version must be bumped in **both** `.claude-plugin/plugin.json` and
`.claude-plugin/marketplace.json` — the plugin cache is keyed by version.
`claude plugin tag .` fails the release if they disagree.

To pick up a new version, `install` is **not** enough — it no-ops with
"already installed". Use `update`, and give the full `plugin@marketplace` id
(the bare plugin name reports "not found"):

```bash
claude plugin marketplace update distributed-train-skills
claude plugin update distributed-train@distributed-train-skills
```

## Out of scope for v0.1

No slash commands and no subagent, by design. A `training-config-reviewer`
agent that reads a real repo's `train.py` / DeepSpeed / FSDP configs is the
strongest future candidate, but it should only be added after the six skills'
trigger rates are measured — introducing a new competitor while the
descriptions are still being tuned would confound both.

## License

MIT
