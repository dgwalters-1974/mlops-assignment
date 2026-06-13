# Phase 1 — Serving configuration

## Workload framing

** For all the following, vLLM 0.22.1 was used rather than 0.9-0.10 recommended in the docs - seemed to be an incompatibility between Qwen's tokenizer and the model.

The agent makes ~3 sequential vLLM calls per user request. The platform SLO is P95 end-to-end < 5 s at 10+ RPS sustained, all of which gives us: a **per-call latency budget of ~1.5 s** (5 s ÷ 3 calls, with a small buffer for sqlite + Python overhead) and a **concurrency target of ~50 in-flight vLLM calls**. The choices below are designed to roughly optimise these two facts on one 80 GB H100.

The model architecture (verified against [`config.json`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json)) sharpens these: 31 B params total / 3.3 B active per token (8-of-128 expert routing), 48 layers, GQA with 4 KV heads × head_dim 128, 256 K native context. MoE makes decode compute cheap (by a factor of roughly 1/10), but all 31 B parameters must still live in VRAM (router decides per-token at inference). Per-token KV cost is exactly 96 KB at bf16 / 48 KB at fp8 — combined with the 50-concurrent target, this is what makes some flags below structurally required, not optional.

## Configuration and rationale

https://docs.vllm.ai/en/latest/configuration/engine_args/?h=argument (old link moved)

**`--quantization fp8`** — bf16 weights take 62 GB of 80 GB, leaving ~6 GB for KV at default 0.9 utilization (`0.9 × 80 − 62 − ~4 GB workspace`); nowhere near enough for 50 concurrent, so bf16 makes the SLO infeasible. fp8 halves weights to 31 GB and runs at ~2× speed on H100's native fp8 tensor cores. Syntax is old school but works, specification has been changed to 'fp8_per_tensor'.

**`--kv-cache-dtype fp8`** — fp8 weights alone leave ~0 GB of slack at peak 50 concurrent; any burst triggers vLLM evicting running sequences, blowing P99 values (preemption). fp8 KV halves per-token KV (96 KB -> 48 KB), restoring ~19 GB of slack and raising the concurrency ceiling from ~50 to ~100.

**`--max-model-len 8192`** — Ceiling 256 K is irrelevant when real prompts are ≤3 K. Capping at 8 K shrinks vLLM's worst-case-per-request provisioning by exactly 32× (262,144 / 8192), so the admission planner fits 50 concurrent: at fp8 KV, 50 × 8192 × 48 KB ≈ 20 GB (fits in the ~39 GB pool) vs 50 × 262 144 × 48 KB = 630 GB at native. 8192 is the smallest power-of-2 above worst case real prompt.

**`--max-num-seqs 64`** — scheduler-level cap on running-batch size, independent of KV. Below 50 makes the SLO impossible; far above the KV-derived ceiling (~100 with fp8 KV) wastes scheduler overhead. 64 = 50 target + ~28% burst headroom, rounded to power of 2.

**`--max-num-batched-tokens 2048`** — at 50 concurrent decoders with new ~3 K prefills arriving every ~33 ms, an unchunked prefill stalls every decode in its batch step — head-of-line blocking that spikes P99. 2048 splits 3 K prefills into 2 chunks that interleave with decode work. Trades slight prefill slowdown for P95 stability.

**`--gpu-memory-utilization 0.92`** — dedicated H100, no competing tenants, so a modest bump from default 0.9 buys ~1.6 GB more KV pool (~3–4 more concurrent requests' headroom). Increment is small because the failure mode is binary.

**`--enable-prefix-caching`** — verify/revise calls share a ~1900-token prefix (system + schema + question) with generate_sql; across requests on the same DB the ~1500-token schema is identical. Cache hits -> ~10–20× prefill speedup on calls 2 and 3, zero quality cost. Deterministic prefix guaranteed by `@lru_cache`-stable schema rendering in `agent/schema.py`. Single biggest one-flag P95 lever for this workload.

**`--enable-chunked-prefill`** — companion to `--max-num-batched-tokens`; without it the per-step token budget can't be enforced across long prefills and head-of-line blocking returns. Default-on in current vLLM.

## Risks and detection

- **fp8 weight + fp8 KV quality regression** — will test on BIRD. If deemed too inaccurate then we might have to fall back to bf16 and accept lower load slack, potentially lower stability.
- **OOM at 0.92 utilization** — load test may need dropping down to 0.9 but worth chancing this higher value as it may give us much needed headroom.
- **Prefill starvation at `max-num-batched-tokens 2048`** — detect via Grafana showing prefill queue growth while decode steps go underused; mitigate by bumping to 4096. Tweaking balance between decoder / prefil will be clearer only during experimentation but this is as good a starting point as any.

# Phase 2 — O11y Core

Dashboard created in Grafana saved with Share->Export->Save to File and then persisted via mounted 'infra/grafana/provisioning/' within Docker and available via dashboards/dashboards.yaml and serving.json. Shows latency, throughput and KV cache.

**Row 1 — concurrency and headroom.** `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`. Together they tell you whether vLLM has slack for more load: queue depth > 0 sustained = admission backpressure; KV > 90% = preemptions imminent, P99 about to blow.

**Row 2 — latency, where time is spent.** `vllm:e2e_request_latency_seconds_bucket` and `vllm:time_to_first_token_seconds_bucket` rendered as P50/P95/P99 via `histogram_quantile()` over a 5m sliding window. Splitting E2E from TTFT is the key diagnostic — if TTFT is fine but E2E is bad, the slowness is in generation, not prefill or queueing.

**Row 3 — throughput.** `rate(vllm:generation_tokens_total[1m])` and `rate(vllm:prompt_tokens_total[1m])` as token rates by direction, plus `rate(vllm:request_success_total[1m])` for completed RPS. Pair with row 2 to read effective sustained capacity (the SLO number is P95 latency at sustained RPS).

**Row 4 — leverage signals.** `vllm:num_preemptions_total` (any growth = KV pressure is binding, expect P99 spikes), and prefix cache hit rate as `rate(vllm:prefix_cache_hits_total[5m]) / rate(vllm:prefix_cache_queries_total[5m])`. The hit-rate panel is the empirical check on Phase 1's prefix-caching bet — climbed to 80%+ during both the baseline eval and the load test, as predicted.

The dashboard earned its keep in Phase 6: KV cache at 0% and queue depth at 0 under sustained 10 RPS clearly highlighted the fact that bottleneck was with the agent rather than vLLM.

Dashboard reaction saved in `screenshots/grafana_serving.png` and dashboard at `infra/grafana/provisioning/dashboards/serving.json`.

# Phase 3 — Agent

"A question that triggered a productive revise".

See .json output below. "Mention the reputation of users who had obtained the badge on 7/19/2010 7:39:08 PM" initially (iter 1 + 2) the loop tried to match 'Date' with a date string but returned no rows both times triggering 'zero rows where rows are implied' and a third iter which used 'Date' LIKE datestring and matched the gold SQL response.

```json
{
  "question": "Mention the reputation of users who had obtained the badge on 7/19/2010 7:39:08 PM.",
  "db_id": "codebase_community",
  "gold_sql": "SELECT T1.Reputation FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T2.Date = '2010-07-19 19:39:08.0'",
  "agent_ok": true,
  "agent_error": null,
  "gold_ok": true,
  "gold_error": null,
  "iterations_used": 3,
  "final_sql": "SELECT u.\"Reputation\"\nFROM \"users\" u\nJOIN \"badges\" b ON u.\"Id\" = b.\"UserId\"\nWHERE b.\"Date\" LIKE '2010-07-19 19:39:08%';",
  "final_correct": true,
  "per_iteration": [
    {
      "iteration": 1,
      "sql": "SELECT u.\"Reputation\"\nFROM \"users\" u\nJOIN \"badges\" b ON u.\"Id\" = b.\"UserId\"\nWHERE b.\"Date\" = '2010-07-19 19:39:08';",
      "correct": false,
      "exec_ok": true,
      "error": null
    },
    {
      "iteration": 2,
      "sql": "SELECT u.\"Reputation\"\nFROM \"users\" u\nJOIN \"badges\" b ON u.\"Id\" = b.\"UserId\"\nWHERE b.\"Date\" = '2010-07-19 19:39:08';",
      "correct": false,
      "exec_ok": true,
      "error": null
    },
    {
      "iteration": 3,
      "sql": "SELECT u.\"Reputation\"\nFROM \"users\" u\nJOIN \"badges\" b ON u.\"Id\" = b.\"UserId\"\nWHERE b.\"Date\" LIKE '2010-07-19 19:39:08%';",
      "correct": true,
      "exec_ok": true,
      "error": null
    }
  ],
  "wall_clock_seconds": 1.5597036950002803
}
```

There were also cases where the loop worked the 'wrong' way - where the first iteration produced the correct SQL but verify flagged it anyway. The loop then produced a different version of the SQL which was scored incorrect. This can be seen below:

```json
{
  "question": "Please list the name of the cards in the set Coldsnap with the highest converted mana cost.",
  "db_id": "card_games",
  "gold_sql": "SELECT T1.name FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T2.name = 'Coldsnap' ORDER BY T1.convertedManaCost DESC LIMIT 1",
  "agent_ok": true,
  "agent_error": null,
  "gold_ok": true,
  "gold_error": null,
  "iterations_used": 2,
  "final_sql": "SELECT c.\"name\"\nFROM \"cards\" c\nJOIN \"sets\" s ON c.\"setCode\" = s.\"code\"\nWHERE s.\"name\" = 'Coldsnap'\n  AND c.\"convertedManaCost\" = (\n    SELECT MAX(c2.\"convertedManaCost\")\n    FROM \"cards\" c2\n    JOIN \"sets\" s2 ON c2.\"setCode\" = s2.\"code\"\n    WHERE s2.\"name\" = 'Coldsnap'\n  );",
  "final_correct": false,
  "per_iteration": [
    {
      "iteration": 1,
      "sql": "SELECT c.\"name\"\nFROM \"cards\" c\nJOIN \"sets\" s ON c.\"setCode\" = s.\"code\"\nWHERE s.\"name\" = 'Coldsnap'\nORDER BY c.\"convertedManaCost\" DESC\nLIMIT 1;",
      "correct": true,
      "exec_ok": true,
      "error": null
    },
    {
      "iteration": 2,
      "sql": "SELECT c.\"name\"\nFROM \"cards\" c\nJOIN \"sets\" s ON c.\"setCode\" = s.\"code\"\nWHERE s.\"name\" = 'Coldsnap'\n  AND c.\"convertedManaCost\" = (\n    SELECT MAX(c2.\"convertedManaCost\")\n    FROM \"cards\" c2\n    JOIN \"sets\" s2 ON c2.\"setCode\" = s2.\"code\"\n    WHERE s2.\"name\" = 'Coldsnap'\n  );",
      "correct": false,
      "exec_ok": true,
      "error": null
    }
  ],
  "wall_clock_seconds": 1.4469292310004676
}
```

Notice the wall clock time for both these requests was in line with that we expected.

# Phase 4 — Agent o11y

Langfuse runs locally from `docker-compose.yml` on port 3001 (the `LANGFUSE_INIT_ORG_ID` and `LANGFUSE_INIT_PROJECT_ID` env vars are required — if either is missing, every other init var gets silently ignored, which caught me out once). The agent wires it up in `agent/server.py` by attaching a `CallbackHandler()` to `graph.invoke(state, config={"callbacks": [handler]})` — LangGraph then emits a span per node and the handler ships them to Langfuse with prompts, completions, latency, and token counts attached. A typical 3-iteration trace shows the full `generate_sql → execute -> verify -> revise -> execute -> verify -> revise -> execute -> verify` waterfall (`screenshots/langfuse_trace.png`), with the vLLM calls dominating wall-clock while `execute` and `verify` are tiny by comparison. Each request from the eval and load drivers is tagged with `{"experiment": "eval-baseline" | "eval-after-tuning", "db_id": "<…>"}`, passed through as LangChain `metadata=` rather than Langfuse `tags` — a small API quirk worth flagging since the two aren't quite interchangeable. Tags visible in `screenshots/langfuse_tags.png`. The intended Phase 6 use was to filter on `experiment=eval-after-tuning` and diff span-level latencies against baseline to localise where the vLLM tuning landed; in practice the delta was within noise so the diff didn't reveal much, but the tagging is in place for any future tuning round.

Images of langfuse output saved at: `/screenshots/langfuse_tags.png` and `/screenshots/langfuse_trace.png`

# Phase 5 - Evals

Agent's final SQL compared with gold SQL on target DB - same rows returned = same answer. Once iteration is correct then it remains correct to avoid punishing loop unfairly. Overall pass rate with this methodology was 30%, with passes evenly distributed at iter 1 / 2 / 3 (30%/27%/30%).

67% of questions resolve after iter 1 (verifier accepted or didn't trip).

`formula_1`, `toxicology`, `thrombosis_prediction` and `card_games` all scored 0% probably due to the non specific column names and domain specific terminology.

'Did the loop do real work?' - Not at this sample size. The revisions within the loop were at times harmful, at times beneficial so difficult to get too positive on its efficacy:

`Productive revise` — codebase_community Q28. Badge timestamp '2010-07-19 19:39:08' returned zero rows; the DB actually stores '…:08.0'. Verify caught the empty result, iter 3 swapped = for LIKE … %, scored correct.

`Harmful revise` — card_games Q15. Iter 1 already had a correct ORDER BY … DESC LIMIT 1, verify flagged it anyway, iter 2 rewrote to a MAX() subquery that returned a different row set and scored wrong.

# Phase 6 - SLO

**P95 end-to-end agent latency under 5 seconds, 10+ RPS (1rps = 1 full agent run per second) over a 5-minute window.**

*(Ahead of running the analysis, What will Phase 6 will likely revise?*

*Starting config derived from SLO sketch, not observed load. Most likely to iterate: `--max-num-batched-tokens` (prefill/decode balance is the most workload-sensitive lever), `--gpu-memory-utilization` (push higher if no OOM and KV pressure remains binding), `--max-num-seqs` (raise if scheduler cap binds before KV).)*

Baseline (phase 1 config):

```json
"summary": {
    "requested_rps": 10.0,
    "duration_seconds": 300,
    "wall_clock_seconds": 360.02306494799996,
    "total_requests": 3000,
    "achieved_rps": 8.332799456705104,
    "ok": 2613,
    "timeouts": 3,
    "http_errors": 383,
    "client_errors": 1,
    "latency_p50": 1.750134606998472,
    "latency_p95": 10.659349872001258,
    "latency_p99": 16.37752916700083,
    "latency_max": 54.78325695100102
}
```

Requests are piling up the system (8.33 vs. 10 RPS) as they are not being cleared quickly enough. This is likely because a third of requests run the full 3 iteration sequence so our initial 'maths' was likely wrong.

*What the dashboard said:*

- Requests running: 20–35 concurrent ✓
- Queue depth: 0 — no admission backpressure
- KV cache: 0% — vLLM has huge unused headroom
- vLLM E2E P95: ~2 s — vLLM itself is healthy at the call level
- TTFT P95: ~200 ms — prefill cheap
- Prefix cache hit rate: 80-90%
- Preemptions: 0

So with vLLM healthy (kv @ 0%, queue @ 0, no preemptions), but per-call P95 so large at this concurrency I changed 'max-num-batched-tokens' parameter from 2048->4096 in order to reduce competition for the per step budget so the prefil stalls fewer decode steps. I then ran the same profile again.

|                       | Before          | After           | Δ           |
|-----------------------|-----------------|-----------------|-------------|
| Achieved RPS          | 8.33            | 8.33            | —           |
| OK / 500 / timeout    | 2613 / 383 / 3  | 2614 / 384 / 1  | flat        |
| P50                   | 1.75 s          | 1.68 s          | −4%         |
| P95                   | 10.66 s         | 10.42 s         | −2% (noise) |
| P99                   | 16.4 s          | 17.7 s          | +8% worse   |
| Max                   | 54.8 s          | 40.9 s          | −25%        |

See `screenshots/grafana_after.png`. The targeted vLLM metric (per-call latency) barely moved; P99 regressed slightly; P95 stayed firmly past SLO - the approach was wrong.

KV at 0% and queue at 0 already told us vLLM had headroom. The bottleneck is agent-side serialisation: 3 sequential vLLM calls * up to 3 revise iterations = up to 9 round-trips per user request.

I think overall, the diagnosis was right, the lever was wrong. Real movers given more time would be (1) reducing MAX_ITERATIONS 3→2 (cuts worst-case fan-out 33%), (2) running multiple uvicorn workers on the agent (parallelise sequential agent calls across requests), (3) fixing the schema-render FK crash that's eating 12.8% of throughput on retries.

### Quality survived

Re-ran the eval against the tuned vLLM — `results/eval_after_tuning.json`:

|                       | Baseline       | After-tune              |
|-----------------------|----------------|-------------------------|
| Overall pass rate     | 30%            | 30%                     |
| Per-iter 1/2/3        | 30 / 27 / 30   | 30 / 30 / 30            |
| Per-DB rates          | (9 DBs)        | bit-for-bit identical   |

Expected — vLLM throughput tuning doesn't touch agent correctness.

### Verdict

SLO missed. P95 = 10.42 s vs 5 s target — ~2x over budget, ~12% error rate on top. Phase 1 config was already over-provisioned for vLLM (KV cache 0% under load); the gap is structural to the agent's sequential call graph, not the inference layer.

# Phase 6 (part 2)

Nebius tenant started crashing during the Phase 6 experiments — instances were vanishing mid-session and I lost work twice — so once I'd burned through my original window I picked up the rest of the iterations on a RunPod H100 SXM. Same workload, same Phase 1 config as the iter 1 baseline, same 10 RPS / 5 min load profile. Three quirks worth flagging up front:

- RunPod pods are themselves Docker containers and don't allow Docker-in-Docker, so I couldn't bring up the Grafana / Prometheus / Langfuse stack via the project's `docker-compose.yml`. I worked from the load test JSON numbers as the evidence rather than capturing fresh dashboard screenshots. The dashboard config (`serving.json`) is unchanged so the framing from part 1 carries over conceptually.
- vLLM had to be bumped past the 0.10 pin again (same Qwen2Tokenizer issue as session 1), and DeepGEMM had to be disabled via `VLLM_USE_DEEP_GEMM=0` because the package wouldn't build on the pod.
- The first attempt at iter 2 below drowned in Langfuse retry queues — the agent was trying to export traces to `localhost:3001` but RunPod's nginx was holding that port and returning 405 to every export attempt. Stripping the Langfuse env vars from `.env` fixed it. Worth knowing if anyone repeats this on RunPod.

With that context, here's the iteration log. Iter 1 in part 1 is the original `max-num-batched-tokens 2048 → 4096` change.

### Iter 2 — `MAX_ITERATIONS 3 → 2` (agent side)

- **Saw:** at iter 1, the achieved 8.33 RPS vs 10 requested showed requests piling up at the agent; about a third of questions used all 3 revise iterations, so worst-case fan-out is 9 vLLM calls per request.
- **Hypothesised:** capping at 2 cuts worst-case fan-out by a third and should compress P95.
- **Changed:** `MAX_ITERATIONS = 3 → 2` in `agent/graph.py`. First attempt drowned in Langfuse retry queues (see above) — redid it after stripping the env vars.
- **Result:** wrong metric moved. P95 went 10.42 → 13.09 s (slightly worse), but achieved RPS climbed 8.33 → 9.20 and timeouts dropped 1 → 0. So the change is a throughput win, not a P95 win — P95 is bound by per-call latency under contention, not by call count. As the README puts it — "as sometimes a metric improves and the SLO doesn't, which is its own lesson."

I reverted `MAX_ITERATIONS` back to 3 for iters 3-6 so each subsequent test is a clean A/B against the iter 1 baseline.

### Iter 3 — `--gpu-memory-utilization 0.92 → 0.85`

- **Saw:** KV cache at 0% under load — vLLM has lots of headroom on the KV pool.
- **Hypothesised:** shrinking the pool should be a no-op since we're nowhere near filling it.
- **Changed:** `--gpu-memory-utilization 0.92 → 0.85` in `scripts/start_vllm.sh`.
- **Result:** hypothesis falsified. P95 dropped 10.42 → 7.09 s (-32%), P99 17.7 → 10.5 s (-40%), achieved RPS 8.33 → 9.08. Every metric improved meaningfully. But this was the first run after a fresh vLLM restart following the iter 2 disaster, so I couldn't tell whether the improvement was from the flag change or just the restart. Ran a control next.

### Iter 3b — control: reverted to the iter 1 config exactly

- **Saw:** iter 3's improvement was suspiciously good and confounded with a fresh restart.
- **Hypothesised:** if the restart is what helped, running the iter 1 config (0.92, 64, 4096) on a fresh restart should produce numbers close to iter 3.
- **Changed:** reverted `--gpu-memory-utilization 0.85 → 0.92`, fresh restart.
- **Result:** **P95 = 7.64 s**. Confirmed — the fresh restart was doing roughly 70% of the work; the 0.85 flag was only contributing ~7% (within noise). So vLLM appears to develop latency drift across a 5-minute load test that the dashboard doesn't surface. The corrected baseline for the remaining iterations is **7.64 s**, not 10.42 s.

### Iter 4 — `--max-num-seqs 64 → 32`

- **Saw:** dashboard's running-batch panel peaked at 30-35 — at first glance the 64 cap looked over-provisioned.
- **Hypothesised:** dropping the cap to 32 should be roughly neutral since we never hit the higher cap.
- **Changed:** `--max-num-seqs 64 → 32`.
- **Result:** strong regression. P50 1.67 → 15.4 s, P95 7.64 → 26.4 s. At our offered load (10 RPS × 3 vLLM calls each ≈ 30 concurrent steady-state) the 32 cap binds and requests queue up in vLLM's admission scheduler. The dashboard's instantaneous panel hid this — it shows the visible running count, not the pressure at the cap. So iter 1's calibration here was actually right; the dashboard's "headroom" framing was misleading on this particular flag.

### Iter 5 — `--max-num-batched-tokens 4096 → 8192`

- **Saw:** iter 1 already bumped this 2048 → 4096 (null). Wanted to see if more headroom helped further.
- **Hypothesised:** doubling the per-step token budget should be flat or marginally better.
- **Changed:** `--max-num-batched-tokens 4096 → 8192`.
- **Result:** first run cascaded — ~50% of requests succeeded then everything fell off a cliff with a wave of timeouts and disconnects. Same shape as the Langfuse-flood crash from iter 2 but Langfuse was already disabled. Looked like a vLLM stability problem at the new memory pressure rather than a clean measurement. Re-ran to disambiguate (iter 5b).

### Iter 5b — replicate of iter 5

- **Saw:** iter 5 was potentially confounded — either a one-off or 8192 is genuinely unstable.
- **Hypothesised:** if iter 5's failure was transient, the second run should be clean.
- **Changed:** same config, fresh restart.
- **Result:** ran clean. P95 = 6.80 s (~11% better than iter 3b, just below my ±15% noise floor), achieved RPS 9.80. So 8192 *can* be marginally better than 4096 when stable — but with 50/50 catastrophic failure across two runs, the config is sitting right at the stability boundary and isn't worth shipping. Reverted to 4096.

### Iter 6 — `uvicorn --workers 4` on the agent

- **Saw:** after exhausting plausible vLLM-side flags, the dashboard read still pointed at agent-side serialisation — a single uvicorn process pushing each request through 3 sequential vLLM calls.
- **Hypothesised:** running the agent with 4 workers should parallelise across user requests and drop P95 meaningfully.
- **Changed:** restarted agent with `uv run uvicorn agent.server:app --host 0.0.0.0 --port 8002 --workers 4`. vLLM config left at the safe iter 1 values.
- **Result:** P95 = **6.71 s** (12% better than iter 3b, still within noise), P50 1.43 s, achieved RPS 8.45. Small real improvement but much less than I expected. So the agent's 3-call serialisation wasn't actually the dominant constraint at this load — the dominant constraint is per-call vLLM latency × 3 calls, which puts a structural floor around 6-7 s P95 regardless of how many agent workers we run.

### After-tuning eval

Re-ran the eval against the final config (workers=4, MAX_ITERATIONS=3, max-num-batched-tokens=4096) → `results/eval_after_tuning_v2.json`.

|                          | Baseline       | After-tuning v2         |
|--------------------------|----------------|-------------------------|
| Overall pass rate        | 30%            | 30%                     |
| Per-iter 1 / 2 / 3       | 30 / 27 / 30   | 30 / 27 / 30            |
| Iteration distribution   | 20 / 2 / 8     | 20 / 2 / 8              |
| Mean wall clock          | 1.02 s         | 0.97 s                  |
| Per-DB pass rates        | (9 DBs)        | bit-for-bit identical   |

Quality survived across all 6 iterations, which was the expected result — none of these flags or worker counts touch agent correctness, they only affect throughput and tail latency.

### Verdict (part 2)

**SLO still missed**, but with a cleaner read on why. Best stable P95 = **6.71 s** against the 5 s target — gap of ~34%. After 6 iterations the picture is:

- vLLM was correctly calibrated in Phase 1. Three out of three flags I touched either regressed (`max-num-seqs` lower), changed within noise (`gpu-memory-utilization`), or sat at the stability ceiling (`max-num-batched-tokens` higher).
- vLLM develops latency drift across a 5-minute load test that the dashboard doesn't show — a fresh restart clawed back ~3 s of P95. That's an operational lever for a real deployment (periodic restarts), not a one-time tuning win.
- Agent-side workers only contributed ~12%, so the agent's serialisation isn't the dominant constraint at this load either.
- What's left is structural: 3 sequential vLLM calls × ~2 s per-call P95 under contention ≈ a 6-7 s P95 floor. To get under 5 s you'd need to either cut per-call latency (smaller model, speculative decoding) or genuinely parallelise the agent's calls — but verify depends on generate's output so that's not possible without redesigning the loop.

The honest read: the 5 s SLO probably isn't achievable with this exact architecture without compromising elsewhere.

## Phase 7 - Wrap up

**Did the agent loop earn its keep?** Not at this sample size. Per-iteration pass rate was 30 / 27 / 30 — flat. The iter 2 dip is within noise (one question is 3.3% on a sample of 30). 8 of the 30 questions ran all the way to iter 3 so the loop is working, but the wins and losses cancel out — the productive revise on codebase_community Q28 vs. the harmful revise on card_games Q15 in Phase 3 are representative. Root cause is structural: without BIRD's `evidence` field in the prompt (we drop it per the task author's guidance), verify has no information generate didn't already have. It can tell that an answer is wrong (empty rows, schema mismatch) but can't reliably say *what's* wrong. Another participant reported 33% -> 53% with evidence surfaced, so the loop probably tips positive once you let it use the hints.

**What I'd do with more time** — in rough order of expected impact:

1. **Smaller / faster main model** (e.g. Qwen3-7B-Instruct, or speculative decoding via vLLM's `--speculative-config` with a draft model). Phase 6 part 2 landed on a structural P95 floor of ~6-7 s — 3 calls × ~2 s per-call vLLM P95. Halving per-call latency should pull the floor under 5 s. Quality cost is the open question — would need a fresh BIRD eval pass to confirm.
2. **Periodic vLLM restarts as an operational lever.** Iter 3b in part 2 showed ~3 s of P95 drift accumulating across a 5-min load test that the dashboard didn't surface. In a real deployment, restarts every N minutes with a second instance hot-swapping would claw that back. Not a config knob, an ops practice.
3. **Fix the FK rendering crash in `agent/schema.py`** that's behind the 12.8% HTTP 500 rate. Deterministic per DB so an easy patch — I left it alone per the assignment author's guidance but it's eating ~12% of throughput on retries.
4. **Guided decoding for the verify JSON output.** vLLM has a `--guided-decoding-backend` that would replace my defensive parser and its silent `ok=true` fallback on parse failures — verify becomes structurally honest.
5. **fp8 vs bf16 quality A/B** on the same 30 questions. I picked fp8 from first principles in Phase 1 but never actually measured the accuracy cost on this workload.
6. **Per-DB latency breakdown in Grafana.** The dashboard currently lumps all questions together but the long-tail DBs (toxicology, formula_1) dominate P99 and that should be visible.
