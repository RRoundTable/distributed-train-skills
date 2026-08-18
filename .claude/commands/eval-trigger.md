---
description: Run trigger evals for a distributed-train skill, plus an optional head-to-head test against mlops:forge-train. Empirically tests whether the skill description causes Claude to actually invoke it. Usage: /eval-trigger <skill-name>
---

# distributed-train Trigger Eval

Empirically test whether the **$ARGUMENTS** skill description causes Claude to actually invoke it — using an isolated `claude -p` subprocess with only the target skill visible.

## How it works

For each query in `trigger_evals.json`:
1. A stub plugin dir is created containing **only** the target skill's `SKILL.md`
2. `claude -p "<query>"` runs via Python `subprocess.Popen` with `--plugin-dir <stub>` and `--setting-sources ''` — subprocess sees exactly one skill (plus Claude Code built-ins)
3. stream-json is parsed line-by-line; trigger detected via `stream_event/content_block_start` → process killed immediately before skill executes
4. All queries run in parallel via `ThreadPoolExecutor`

**Why the runner script must be executed in the background:**
- `claude -p` deadlocks when its parent process is the blocked Bash tool: the subprocess tries to connect to the parent Claude Code process via `CLAUDE_CODE_SSE_PORT`, but the parent is waiting for the Bash tool to finish
- Running `run.py` in the background (`python3 -u run.py > run.log 2>&1 &`) frees the parent, breaking the deadlock
- The Bash tool then polls `run.log` for completion

**Why `CLAUDECODE`, `ANTHROPIC_API_KEY`, `CLAUDE_CODE_SSE_PORT` must be unset:**
- `CLAUDECODE` — prevents nested Claude Code detection
- `ANTHROPIC_API_KEY` — forces OAuth (claude.ai subscription); Max subscribers without API credits get a silent `billing_error` otherwise
- `CLAUDE_CODE_SSE_PORT` — prevents the subprocess from attempting IPC with the parent

**Why `--setting-sources ''` is needed:**
- Skips global plugin settings; only `--plugin-dir` skills are visible
- Built-in skills (`keybindings-help`, `simplify`, `loop`, `claude-api`) are always present regardless — but these occupy unrelated domains and don't compete with distributed-train skills in practice

## Step 1: Read the eval set

Read:
- `skills/$ARGUMENTS/SKILL.md` — note the `description` field (the only thing Claude sees)
- `skills/$ARGUMENTS/evals/trigger_evals.json` — queries with `should_trigger` labels

## Step 2: Build the stub plugin dir

```bash
mkdir -p /tmp/trigger-eval-$ARGUMENTS/stub/skills/$ARGUMENTS
cp skills/$ARGUMENTS/SKILL.md /tmp/trigger-eval-$ARGUMENTS/stub/skills/$ARGUMENTS/SKILL.md
```

## Step 3: Write the runner script

Write `/tmp/trigger-eval-$ARGUMENTS/run.py`:

```python
#!/usr/bin/env python3
import json, os, select, signal, subprocess, time
from concurrent.futures import ThreadPoolExecutor

SKILL = "$ARGUMENTS"
STUB  = f"/tmp/trigger-eval-{SKILL}/stub"
OUT   = f"/tmp/trigger-eval-{SKILL}/results"
os.makedirs(OUT, exist_ok=True)

UNSET = ("CLAUDECODE", "ANTHROPIC_API_KEY", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_ENTRYPOINT")

def detected(ev):
    if ev.get("type") == "stream_event":
        cb = ev.get("event", {}).get("content_block", {})
        return cb.get("type") == "tool_use" and cb.get("name") == "Skill"
    if ev.get("type") == "assistant":
        return any(c.get("type") == "tool_use" and c.get("name") == "Skill"
                   for c in ev.get("message", {}).get("content", []))
    return False

def kill_proc(p):
    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception: p.kill()
    try: p.wait(timeout=3)
    except Exception: pass

def run_query(item):
    env = {k: v for k, v in os.environ.items() if k not in UNSET}
    p = subprocess.Popen(
        ["claude", "-p", item["query"], "--plugin-dir", STUB,
         "--setting-sources", "", "--output-format", "stream-json",
         "--verbose", "--include-partial-messages"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env=env, start_new_session=True)
    buf = ""
    deadline = time.time() + 30
    try:
        while time.time() < deadline:
            if p.poll() is not None:
                buf += (p.stdout.read() or b"").decode("utf-8", errors="replace")
                break
            if not select.select([p.stdout], [], [], 1.0)[0]:
                continue
            chunk = os.read(p.stdout.fileno(), 8192)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                try:
                    if detected(json.loads(line)):
                        return item["idx"], True
                except Exception:
                    pass
    finally:
        kill_proc(p)
    return item["idx"], False

evals = json.loads(open(f"skills/{SKILL}/evals/trigger_evals.json").read())
items = [{"idx": i, **ev} for i, ev in enumerate(evals)]

with ThreadPoolExecutor(max_workers=10) as pool:
    futures = {pool.submit(run_query, item): item for item in items}
    for future in futures:
        idx, triggered = future.result()
        ev = futures[future]
        open(f"{OUT}/{idx}.json", "w").write(
            json.dumps({"query": ev["query"], "should_trigger": ev["should_trigger"], "triggered": triggered}))

print("done")
```

## Step 4: Run in the background and wait

```bash
python3 -u /tmp/trigger-eval-$ARGUMENTS/run.py > /tmp/trigger-eval-$ARGUMENTS/run.log 2>&1 &
```

Then poll until complete:

```bash
# poll every few seconds
while ! grep -q "done" /tmp/trigger-eval-$ARGUMENTS/run.log 2>/dev/null; do sleep 3; done
cat /tmp/trigger-eval-$ARGUMENTS/run.log
```

## Step 5: Report results

Read all result files and compare `triggered` vs `should_trigger`:

```
$ARGUMENTS — trigger eval (empirical)
──────────────────────────────────────────────────

PASS  [trigger]     just pushed feat(payments): add Stripe webhook...
FAIL  [trigger]     How does the rate limiter work in this project...
PASS  [no-trigger]  can you add pagination to the users endpoint...
...

Results: X/Y  (Z%)
```

For every FAIL:
- State what actually happened (`triggered` / `not_triggered`)
- Hypothesize why the description led Claude to that judgment
- Suggest a specific wording fix in the description

## Step 6: Clean up

```bash
rm -rf /tmp/trigger-eval-$ARGUMENTS
```

## Step 7: Synthesis

After reporting, briefly answer:
- Which queries are edge cases that reveal description ambiguity?
- Is there a pattern in the failures?
- What specific phrase in the description would fix the failures?


---

## Part 2 — head-to-head against `mlops:forge-train` (required for release)

Part 1 isolates **one** skill, so a `should_trigger: false` pass only proves
"this skill stayed quiet". It cannot prove that a cluster-operations query
lands on `mlops:forge-train` instead. That is the crowd-out requirement, and
only this part tests it.

### Step H1 — build the head-to-head stub

One stub dir containing **all six** of our SKILL.md files plus a copy of
forge-train's:

```bash
HH=/tmp/trigger-eval-headtohead
FT=$(find ~/.claude/plugins/marketplaces -path '*/skills/forge-train/SKILL.md' | head -1)
rm -rf $HH && mkdir -p $HH/stub/skills
for s in distributed-training-router parallelism-strategies communication-backends \
         gpu-architecture training-metrics memory-offloading; do
  mkdir -p $HH/stub/skills/$s
  cp skills/$s/SKILL.md $HH/stub/skills/$s/SKILL.md
done
mkdir -p $HH/stub/skills/forge-train
cp "$FT" $HH/stub/skills/forge-train/SKILL.md
ls $HH/stub/skills
```

If `$FT` is empty, the mlops plugin is not installed locally and this part
cannot run — say so rather than skipping it silently.

### Step H2 — the head-to-head query set

Grade on **which skill name appears in the `Skill` tool_use block**, not on a
boolean. Write `$HH/cases.json`:

```json
[
  {"query": "GPU 4장짜리 job 띄워줘",                        "expect": "forge-train"},
  {"query": "job이 QUEUED에서 안 넘어가",                     "expect": "forge-train"},
  {"query": "forge quota 확인",                              "expect": "forge-train"},
  {"query": "방금 실패한 job 로그 봐줘",                       "expect": "forge-train"},
  {"query": "multi-node submit 하는 법",                      "expect": "forge-train"},
  {"query": "내 job 로그에서 NCCL error 확인해줘",             "expect": "forge-train"},
  {"query": "NCCL all-reduce가 ring이랑 tree랑 뭐가 달라?",    "expect": "communication-backends"},
  {"query": "Explain why pipeline parallelism has a bubble",  "expect": "parallelism-strategies"},
  {"query": "A100 80GB에 13B bf16 + Adam 학습 가능해?",        "expect": "memory-offloading"},
  {"query": "What MFU should a 7B model reach on H100?",      "expect": "gpu-architecture"},
  {"query": "GPU 8장 썼는데 5.2배밖에 안 빨라졌어",             "expect": "training-metrics"},
  {"query": "분산학습 처음인데 뭐부터 봐야 해?",                "expect": "distributed-training-router"}
]
```

The `forge-train` rows are the ones that matter most: a real-log OOM query
must **not** be stolen by `memory-offloading`, and a NCCL-error-in-logs query
must **not** be stolen by `communication-backends`.

### Step H3 — runner

Same subprocess mechanics as Part 1 (same `UNSET` list, same background
execution, same `--setting-sources ''`), with one change: instead of returning
a boolean, capture the **`input.command` / skill name** from the `Skill`
tool_use block.

```python
def detect_skill(ev):
    blocks = []
    if ev.get("type") == "assistant":
        blocks = ev.get("message", {}).get("content", [])
    elif ev.get("type") == "stream_event":
        cb = ev.get("event", {}).get("content_block", {})
        if cb:
            blocks = [cb]
    for c in blocks:
        if c.get("type") == "tool_use" and c.get("name") == "Skill":
            inp = c.get("input") or {}
            # partial stream blocks may have empty input; fall back to the
            # assistant-message form which always carries it
            for key in ("skill", "command", "name"):
                if inp.get(key):
                    return str(inp[key])
    return None
```

Because `--include-partial-messages` can deliver a `tool_use` block before its
input is streamed, do **not** kill the process on the first `Skill` block —
keep reading until `input` is populated or the 30 s deadline passes, then kill.
Normalize the returned value by stripping any `plugin:` prefix before
comparing to `expect`.

### Step H4 — report

```
head-to-head vs mlops:forge-train
──────────────────────────────────────────────────
PASS  GPU 4장짜리 job 띄워줘                 → forge-train
FAIL  방금 실패한 job 로그 봐줘              → memory-offloading  (expected forge-train)
...

ops queries routed to forge-train: 5/6
topic queries routed correctly:    6/6
```

### Step H5 — clean up

```bash
rm -rf /tmp/trigger-eval-headtohead
```

## Release gate

- `claude plugin validate . --strict` passes
- both manifests at the same version
- per-skill trigger eval **≥85% overall**
- **100%** on the forge-train ops negatives in each per-skill set
- head-to-head routes **all 6** ops queries to `forge-train`

A failure on the ops negatives is a description bug, not a flake. Fix the
`Do NOT activate for:` clause, do not re-roll the eval.
