---
description: Run behavioral evals for a distributed-train skill on Sonnet 5.0 by default. Measures whether loading the skill actually changes the answer. Usage: /eval-behavior <skill-name> [model-list]
---

# distributed-train Behavioral Eval

Run behavioral evals for the skill named **$ARGUMENTS**.

Trigger evals answer *"does the skill fire?"*. This answers the harder
question: **"once it fires, does it change the answer?"** Every case runs
twice — once with the skill's content injected, once without — and the gap
between them is the only number that matters. A skill whose with- and
without-scores are equal is not earning its context budget, no matter how
well written it is.

## Why the discrimination signal is sharp for this plugin

These are **knowledge** skills, so the assertions are deliberately written
against specific, checkable quantities that a model without the skill will
approximate, hedge, or get wrong:

- exact formulas (`(p−1)/(m+p−1)`, `s·b·h·(34 + 5as/h)`, `2(n−1)S/n`)
- exact figures (H100 dense bf16 ridge point 295, `16Ψ`, ZeRO-3's `3Ψ`)
- verified published numbers (PaLM 46.2% MFU, Llama 3 38–43%)
- the non-obvious inversions (ZeRO-2 is bandwidth-free; the straggler spends
  the *least* time in collectives; offload is a capability tool)

Write new assertions the same way. **"Mentions communication overhead" is a
bad assertion** — a base model says that. **"States ZeRO-2 moves the same 2Ψ
as DDP because AllReduce = ReduceScatter ∘ AllGather" is a good one** — it is
either there or it is not.

## Models to evaluate (default: Sonnet 5.0 only)

By default, behavioral evals run on **`claude-sonnet-5` only** — the model
most users hit, and the discriminating signal we care about per iteration.
Opus is opt-in.

- **Default:** `claude-sonnet-5`
- **Override:** a comma-separated model list as the second argument:
  - `/eval-behavior parallelism-strategies claude-opus-5` → Opus only
  - `/eval-behavior parallelism-strategies claude-opus-5,claude-sonnet-5` → both, in parallel
- Each model writes to `evals/$ARGUMENTS-<suffix>/` where suffix is `sonnet50`,
  `opus50`, `haiku45`. Never collapse two models into one directory.
- Single model → one table. Multiple → side by side with cross-model deltas.

## Setup

Read before proceeding:

- `skills/$ARGUMENTS/SKILL.md` — plus **every** file under
  `skills/$ARGUMENTS/references/`. The runner concatenates them; you should
  know what is in the injected context when you grade.
- `skills/$ARGUMENTS/evals/evals.json` — cases and assertions.

Workspace per model: `evals/$ARGUMENTS-<suffix>/` (create if needed; overwrite
previous runs). This directory is gitignored — it is a scratch artifact.

## Step 1: Write the runner script

Both variants use `claude -p` with `--setting-sources ''` so no CLAUDE.md or
installed plugin leaks into context. The **only** difference between variants
is the prompt: with-skill injects the skill content, without-skill does not.

Write `/tmp/behavior-eval-$ARGUMENTS/run.py`:

```python
#!/usr/bin/env python3
import argparse, json, os, subprocess, signal, tempfile
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--skill", required=True)
ap.add_argument("--model", required=True, help="e.g. claude-sonnet-5")
ap.add_argument("--suffix", required=True, help="output dir suffix, e.g. sonnet50")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--timeout", type=int, default=240)
args = ap.parse_args()

REPO = os.getcwd()
SKILL_DIR = os.path.join(REPO, "skills", args.skill)
OUT = os.path.join(REPO, "evals", f"{args.skill}-{args.suffix}")

# SKILL.md plus every reference file, in stable order
skill_content = open(os.path.join(SKILL_DIR, "SKILL.md")).read()
refs = []
for root, dirs, files in os.walk(SKILL_DIR):
    dirs[:] = [d for d in dirs if d != "evals"]
    for fn in files:
        if fn.endswith(".md") and fn != "SKILL.md":
            path = os.path.join(root, fn)
            refs.append((os.path.relpath(path, SKILL_DIR), path))
for rel, path in sorted(refs):
    skill_content += f"\n\n---\n# {rel}\n\n" + open(path).read()

UNSET = ("CLAUDECODE", "ANTHROPIC_API_KEY", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_ENTRYPOINT")

def kill_proc(p):
    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception: p.kill()
    try: p.wait(timeout=3)
    except Exception: pass

def run_eval(item, with_skill):
    variant = "with_skill" if with_skill else "without_skill"
    out_dir = os.path.join(OUT, item["name"], variant)
    os.makedirs(out_dir, exist_ok=True)

    if with_skill:
        full_prompt = (
            "You are a helpful assistant with the following domain knowledge. "
            "Use ONLY this knowledge to answer. Do NOT run any commands or use tools. "
            "Just provide your answer as text.\n\n"
            "DOMAIN KNOWLEDGE:\n"
            f"{skill_content}\n\n"
            f"TASK: {item['prompt']}\n\n"
            "Provide your complete response based on the domain knowledge above."
        )
    else:
        full_prompt = (
            "Answer this question using only your own general knowledge. "
            "Do NOT read any files or use any tools to look up information. "
            "Just answer directly based on what you know.\n\n"
            f"TASK: {item['prompt']}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        env = {k: v for k, v in os.environ.items() if k not in UNSET}
        try:
            p = subprocess.Popen(
                ["claude", "-p", full_prompt, "--setting-sources", "",
                 "--model", args.model, "--output-format", "text", "--max-turns", "1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                cwd=tmpdir, env=env, start_new_session=True)
            stdout, _ = p.communicate(timeout=args.timeout)
            response = stdout.decode("utf-8", errors="replace").strip()
        except subprocess.TimeoutExpired:
            kill_proc(p); response = "TIMEOUT"
        except Exception as e:
            response = f"ERROR: {e}"

    open(os.path.join(out_dir, "response.txt"), "w").write(response + "\n")
    return item["name"], variant, len(response)

data = json.loads(open(os.path.join(SKILL_DIR, "evals", "evals.json")).read())
tasks = [(ev, ws) for ev in data["evals"] for ws in (True, False)]

with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futs = [pool.submit(run_eval, item, ws) for item, ws in tasks]
    for i, f in enumerate(futs, 1):
        n, v, l = f.result()
        print(f"  [{i}/{len(futs)}] {n} [{v}]: {l} chars", flush=True)
print("done")
```

## Step 2: Run it (background)

`claude -p` deadlocks if its parent is a blocked Bash tool, so launch with
`run_in_background: true` (or `&`) and poll the log.

```bash
mkdir -p /tmp/behavior-eval-$ARGUMENTS
# write the script above, then:

python3 -u /tmp/behavior-eval-$ARGUMENTS/run.py \
  --skill $ARGUMENTS --model claude-sonnet-5 --suffix sonnet50 \
  > /tmp/behavior-eval-$ARGUMENTS/run-sonnet50.log 2>&1 &
```

Multi-model is opt-in only — one runner per model, concurrently, each with its
own suffix and log. Then wait:

```bash
for log in /tmp/behavior-eval-$ARGUMENTS/run-*.log; do
  while ! grep -q "done" "$log" 2>/dev/null; do sleep 5; done
done
tail -n 3 /tmp/behavior-eval-$ARGUMENTS/run-*.log
```

## Step 3: While waiting

Briefly walk the user through what the assertions are testing. Do not repeat
it per model — assertions are model-independent.

## Step 4: Grade

Grade each response against its case's `assertions`, one at a time, citing
specific evidence from the output. Grade **both** variants — the without-skill
score is the baseline that makes the with-skill score meaningful.

For 5+ cases, delegate grading to a `general-purpose` subagent so it does not
consume your context. Give it the `evals.json` path and the response
directory, and ask for one concise table plus per-assertion evidence.

Grading rules that keep the numbers honest:

- An assertion passes only on **explicit** content. "Could be inferred from
  what it said" is a fail — the point is whether the skill supplied it.
- A **wrong** specific number is a fail, not a partial pass. A model
  confidently stating the H100 ridge point is 150 is worse than one that
  omits it.
- Grade the without-skill run **first**, or at least independently. Knowing
  the expected answer makes it very easy to read it into a vague response.

Optionally record machine-readable results at
`evals/$ARGUMENTS-<suffix>/<case>/<variant>/grading.json`:

```json
{
  "assertions": [
    {"id": "zero2-same-volume", "passed": true, "evidence": "one sentence citing the output"}
  ],
  "summary": {"passed": 3, "failed": 0, "total": 3, "pass_rate": 1.0}
}
```

## Step 5: Report

```
$ARGUMENTS — behavioral eval (model=claude-sonnet-5)

  case                          with    without   Δ
  ----------------------------  ------  --------  -----
  zero-stage-comm-volume         3/3     0/3      +100pp
  pipeline-bubble-formula        3/3     1/3       +67pp
  ...

Totals:
  with_skill:     A/B (P%)
  without_skill:  C/D (Q%)
  discrimination: +Δ pp
```

Then call out, in this order:

1. **Regressions** — a case failing *with* the skill. The skill content is
   wrong, missing, or buried. Highest priority; fix the reference file.
2. **Non-discriminating** — passes with **and** without. Either the assertion
   is too lenient (tighten it to a specific number) or the content is common
   knowledge (consider cutting it — it is costing context for nothing).
3. **Strongest effect** — the top 3 cases by delta. These justify the skill;
   keep them stable as regression guards.
4. *(multi-model only)* **Cross-model regressions** — passes on Opus, fails on
   Sonnet. The content is present but not findable by the weaker model; that
   is a structure problem, usually something buried in prose that belongs in
   a table.

## Step 6: Clean up

```bash
rm -rf /tmp/behavior-eval-$ARGUMENTS
```

Keep `evals/$ARGUMENTS-<suffix>/` for inspection. It is gitignored, not
committed.

## Relationship to the trigger eval

| | `/eval-trigger` | `/eval-behavior` |
|---|---|---|
| Question | does the description fire? | does the content change the answer? |
| Reads | `SKILL.md` frontmatter only | `SKILL.md` + all `references/` |
| Fix location | the `description:` field | the reference files |
| Failure means | routing bug | content bug |

Both must pass. A skill that triggers perfectly and changes nothing is
overhead; a skill with excellent content that never fires is unreachable.
