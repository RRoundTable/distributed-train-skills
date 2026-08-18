# Symptom Index

Observation → budget → owning skill, with the check that discriminates it
from its nearest neighbour. Symbols: `notation-and-glossary.md`. Budgets:
compute / capacity / bandwidth / serialization (router SKILL.md).

> Platform recipe: if the observation comes from a **real job on a cluster** —
> a job log, an exit code, a scheduler state, a launcher error — route to skill
> `mlops:forge-train` first. This index is for interpreting mechanisms, not
> for retrieving or acting on job state.

## Errors and crashes

| You saw | Budget | Skill | Discriminating check |
|---|---|---|---|
| `torch.cuda.OutOfMemoryError: Tried to allocate …` | capacity | memory-offloading | *When* in the step? before step 1 → model states; mid-forward → activations; at `optimizer.step()` → optimizer state / fp32 master copy |
| OOM with `reserved memory is much greater than allocated` | capacity | memory-offloading | fragmentation, not a real shortage — see the allocator playbook |
| OOM only after N successful steps | capacity | memory-offloading | growth: retained graph, a list of tensors, eval batch larger than train, or fragmentation accumulating |
| `Watchdog caught collective operation timeout` | serialization | communication-backends | a collective is a barrier — some rank never arrived. Find *which*, don't raise the timeout |
| `NCCL WARN … unhandled system error` / bootstrap failure | — | **mlops:forge-train** | this is fabric/interface configuration on a specific cluster |
| Hangs at the first collective, no error | serialization | communication-backends | rank/world-size mismatch, mismatched collective order, or a rank that crashed silently |
| Works on 1 node, hangs on 2 | serialization | communication-backends | intra- vs inter-node path differs; also a classic mismatched-shape all-reduce |
| Exit 137 / killed | — | **mlops:forge-train** | host OOM or the scheduler, not a GPU-side condition |
| `CUDA error: device-side assert` | compute | gpu-architecture | usually an index out of range (labels ≥ `V`); re-run on CPU to localize |
| `Expected all tensors on the same device` after adding parallelism | — | parallelism-strategies | a module not placed on the mesh |

## Throughput and utilization

| You saw | Budget | Skill | Discriminating check |
|---|---|---|---|
| MFU < 20% | measure first | training-metrics | confirm the FLOP count before diagnosing; three common inflations/deflations are catalogued there |
| MFU low, GPU util ~100% | compute | gpu-architecture | you are running kernels, they are just inefficient — roofline the dominant op |
| MFU low, GPU util low, no OOM, **stable** step time | bandwidth | communication-backends | profile shows time inside NCCL kernels |
| MFU low, GPU util low, **variable** step time | serialization | communication-backends | p99 ≫ p50 → straggler; regular sawtooth → pipeline bubble |
| 8 GPUs → 5.2× | bandwidth | training-metrics → communication-backends | strong or weak scaling? then compute-to-comm ratio `6·b·m·s / bytes_per_elem` |
| Scaling fine within a node, falls off across nodes | bandwidth | communication-backends | the NVLink→fabric step is ~1/4 to 1/67; check the parallelism *ordering* |
| Throughput drops when sequence length doubles | compute | gpu-architecture | attention is `O(s²)`; it crosses 50% of FLOPs when `s > 6h` |
| Throughput fell after enabling gradient checkpointing | compute (paid deliberately) | memory-offloading | expected: full recompute costs ~25–33% of step time |
| First step is 10× slower than the rest | — | gpu-architecture | autotuning / compile / allocator warm-up. Exclude warm-up steps from every benchmark |
| Step time creeps upward over hours | capacity | memory-offloading | allocator fragmentation, or a leak; check `reserved` vs `allocated` trend |
| Step time creeps upward but forward/backward/optimizer times are each **flat** | serialization | training-metrics → communication-backends | not fragmentation — rank *arrival* times at collectives are drifting apart; measure the spread, suspect per-process work on the critical path (GC) |
| Same config benchmarked twice, differs by several % | serialization | communication-backends | placement varies run to run; a ~10%-slow host anywhere in the job sets the pace |
| Minutes of dead time between launch and step 1, worse at larger `n` | serialization | communication-backends | rendezvous / process-group init scaling; invisible in any per-step metric |
| Vocabulary-size change moved throughput several % | compute | gpu-architecture | tile granularity — pad `V` to a multiple of 64/128 |

## Loss and correctness

| You saw | Budget | Skill | Discriminating check |
|---|---|---|---|
| Loss changed when I changed the parallelism config | correctness | training-metrics | is `B = b·m·d` identical? mean-of-means with unequal token counts? per-rank RNG seeding? |
| Loss spiked and recovered | — | training-metrics | usually data or a rare long sample; check the batch at that step |
| Loss spiked and diverged | — | training-metrics | logit growth / attention entropy collapse; the small-scale-proxy work (arXiv:2309.14322) reproduces both |
| Loss is NaN in fp16 but fine in bf16 | — | gpu-architecture | fp16 range, not a bug in the parallelism; loss scaling or switch to bf16 |
| Gradient norm spikes on exactly one rank | correctness | training-metrics | data skew or a bad shard, not a communication issue |
| Loss differs run-to-run with the same seed | correctness | training-metrics | nondeterministic reductions; also check dataloader worker seeding |
| Validation loss fine, checkpoint reload is worse | correctness | memory-offloading | sharded-state-dict save/load mismatch under ZeRO-3/FSDP |

## Profiles and traces

| You saw | Budget | Skill | Discriminating check |
|---|---|---|---|
| Large gaps between kernels on the timeline | serialization | communication-backends | CPU-bound launch, or a blocking sync (`.item()`, `.cpu()`) inside the loop |
| NCCL kernels serialized with compute, not overlapped | bandwidth | communication-backends | bucketing and the backward-overlap conditions |
| Many tiny kernels | compute | gpu-architecture | fusion opportunity; also raises launch overhead |
| Kernel achieves a small fraction of peak, high DRAM throughput | compute (memory-bound) | gpu-architecture | it is on the memory roof — its arithmetic intensity is below the ridge point |
| Regular idle blocks at the start and end of each step | serialization | parallelism-strategies | pipeline bubble, fraction `(p−1)/(m+p−1)` |
| One rank's kernels always start late | serialization | communication-backends | straggler; check clocks/thermals and the per-rank input length distribution |
| `Memcpy DtoH` inside the training loop | bandwidth | memory-offloading | a host sync, or an offload path you did not intend |

## Configuration questions phrased as symptoms

| You said | Skill |
|---|---|
| "Will 13B bf16 + Adam fit on an 80GB card?" | memory-offloading |
| "Should I use FSDP or DeepSpeed ZeRO-3?" | parallelism-strategies |
| "TP=8 or TP=4 with PP=2?" | parallelism-strategies |
| "Is ring or tree all-reduce better here?" | communication-backends |
| "What MFU should I expect?" | training-metrics (then gpu-architecture for the peak) |
| "Why is FlashAttention faster?" | gpu-architecture |
| "Can I offload the optimizer to CPU?" | memory-offloading |
| "How often should I checkpoint?" | communication-backends (and first: how expensive is one?) |
| "The job keeps dying — how do I keep it running?" | communication-backends (goodput = MFU × ETTR) |
| "How do I submit this on the cluster?" | **mlops:forge-train** |

## How to use an entry

1. Confirm the observation is real and reproducible — one step's number is
   noise. `training-metrics/references/benchmarking-methodology.md` has the
   minimum protocol.
2. Run the discriminating check *before* loading the skill. Half of these
   rows separate two very different causes, and the check is cheaper than the
   wrong fix.
3. Load the owning skill and use its references for the mechanism and the
   priced fix.
