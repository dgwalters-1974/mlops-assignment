# Phase 1 — Serving configuration

## Workload framing

** For all the following, vLLM 0.22.1 was used rather than 0.9-0.10 recommended in the docs - seemed to be an incompatibility between Qwen's tokenizer and the model.

The agent makes ~3 sequential vLLM calls per user request. The platform SLO is P95 end-to-end < 5 s at 10+ RPS sustained, all of which gives us: a **per-call latency budget of ~1.5 s** (5 s ÷ 3 calls, with a small buffer for sqlite + Python overhead) and a **concurrency target of ~50 in-flight vLLM calls**. The choices below are designed to roughly optimise these two facts on one 80 GB H100.

The model architecture (verified against [`config.json`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json)) sharpens these: 31 B params total / 3.3 B active per token (8-of-128 expert routing), 48 layers, GQA with 4 KV heads × head_dim 128, 256 K native context. MoE makes decode compute cheap (by a factor of roughly 1/10), but all 31 B parameters must still live in VRAM (router decides per-token at inference). Per-token KV cost is exactly 96 KB at bf16 / 48 KB at fp8 — combined with the 50-concurrent target, this is what makes some flags below structurally required, not optional.

## Configuration and rationale:
https://docs.vllm.ai/en/latest/configuration/engine_args/?h=argument (old link moved)

**`--quantization fp8`** — bf16 weights take 62 GB of 80 GB, leaving ~6 GB for KV at default 0.9 utilization (`0.9 × 80 − 62 − ~4 GB workspace`); nowhere near enough for 50 concurrent, so bf16 makes the SLO infeasible. fp8 halves weights to 31 GB and runs at ~2× speed on H100's native fp8 tensor cores. Syntax is old school but works, specification has been changed to'fp8_per_tensor'.

**`--kv-cache-dtype fp8`** — fp8 weights alone leave ~0 GB of slack at peak 50 concurrent; any burst triggers vLLM evicting running sequences, blowing P99 values (preemption). fp8 KV halves per-token KV (96 KB -> 48 KB), restoring ~19 GB of slack and raising the concurrency ceiling from ~50 to ~100.

**`--max-model-len 8192`** — Ceiling 256 K is irrelevant when real prompts are ≤3 K. Capping at 8 K shrinks vLLM's worst-case-per-request provisioning by exactly 32× (262,144 / 8192), so the admission planner fits 50 concurrent: at fp8 KV, 50 × 8192 × 48 KB ≈ 20 GB (fits in the ~39 GB pool) vs 50 × 262 144 × 48 KB = 630 GB at native. 8192 is the smallest power-of-2 above worst case real prompt.

**`--max-num-seqs 64`** — scheduler-level cap on running-batch size, independent of KV. Below 50 makes the SLO impossible; far above the KV-derived ceiling (~100 with fp8 KV) wastes scheduler overhead. 64 = 50 target + ~28% burst headroom,rounded to power of 2.

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

**Row 1 — concurrency and headroom.**             
  `vllm:num_requests_running`,
  `vllm:num_requests_waiting`,                      
  `vllm:kv_cache_usage_perc`. Together they tell you
  whether vLLM has slack for more load: queue depth
  > 0 sustained = admission backpressure; KV > 90%
  = preemptions imminent, P99 about to blow.      
                                         
**Row 2 — latency, where time is spent.**         
`vllm:e2e_request_latency_seconds_bucket` and
`vllm:time_to_first_token_seconds_bucket` rendered
  as P50/P95/P99 via `histogram_quantile()` over a
5m sliding window. Splitting E2E from TTFT is the
key diagnostic — if TTFT is fine but E2E is bad,
the slowness is in generation, not prefill or
queueing.                         
                                  
**Row 3 — throughput.**                           
`rate(vllm:generation_tokens_total[1m])` and
`rate(vllm:prompt_tokens_total[1m])` as token     
rates by direction, plus               
`rate(vllm:request_success_total[1m])` for      
completed RPS. Pair with row 2 to read effective
sustained capacity (the SLO number is P95 latency
at sustained RPS).                
                                    
**Row 4 — leverage signals.**                     
`vllm:num_preemptions_total` (any growth = KV
pressure is binding, expect P99 spikes), and      
prefix cache hit rate as               
`rate(vllm:prefix_cache_hits_total[5m]) /       
rate(vllm:prefix_cache_queries_total[5m])`. The
hit-rate panel is the empirical check on Phase 1's
  prefix-caching bet — climbed to 80%+ during both
the baseline eval and the load test, as predicted.

The dashboard earned its keep in Phase 6: KV cache
at 0% and queue depth at 0 under sustained 10 RPS
clearly highlighted the fact that bottleneck was with the agent rather than vLLM.

Dashboard reaction saved in `screenshots/grafana_serving.png` and dashboard at `infra/grafana/provisioning/dashboards/serving.json`.

# Phase 3 — Agent
"A question that triggered a productive revise".

See .json output below. "Mention the reputation of users who had obtained the badge on 7/19/2010 7:39:08 PM" initially (iter 1 + 2) the loop tried to match 'Date' with a date string but returned no rows both times triggering 'zero rows where rows are implied' and a third iter which used 'Date' LIKE datestring and matched the gold SQL response.
```
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

```
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

# Phase 4 — Agent

Images of langfuse output saved at: `/screenshots/langfuse_tags.png` and `/screenshots/langfuse_trace.png`

# Phase 5 - Evals

Agent's final SQL compared with gold SQL on target DB - same rows returned = same answer. Once iteration is correct then it remains correct to avoid punishing loop unfairly. Overall pass rate with this methodology was 30%, with passes evenly distributed at iter 1 / 2 / 3 (30%/27%/30%).

67% of questions resolve after iter 1 (verifier accepted or didn't trip) 

`formula_1`, `toxicology`, `thrombosis_prediction` and `card_games` all scored 0% probably due to the non specific column names and domain specific terminology.

'Did the loop do real work?' - Not at this sample size. The revisions within the loop were at times harmful, at times beneficial so difficult to get too positive on its efficacy:

`Productive revise` — codebase_community Q28. Badge timestamp '2010-07-19 19:39:08' returned zero rows; the DB actually stores '…:08.0'. Verify caught the
empty result, iter 3 swapped = for LIKE … %, scored correct.                   
`Harmful revise` — card_games Q15. Iter 1 already had a correct ORDER BY … DESC LIMIT 1, verify flagged it anyway, iter 2 rewrote to a MAX() subquery that      
returned a different row set and scored wrong.


# Phase 6 - SLO

**P95 end-to-end agent latency under 5 seconds, 10+ RPS (1rps = 1 full agent run per second) over a 5-minute window.**

*(Ahead of running the analysis, What will Phase 6 will likely revise?*

*Starting config derived from SLO sketch, not observed load. Most likely to iterate: `--max-num-batched-tokens` (prefill/decode balance is the most workload-sensitive lever), `--gpu-memory-utilization` (push higher if no OOM and KV pressure remains binding), `--max-num-seqs` (raise if scheduler cap binds before KV).)*

Baseline (phase 1 config): 

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
Requests are piling up the system (8.33 vs. 10 RPS) as they are not being cleared quickly enough. This is likely because a third of requests run the full 3 iteration sequence so our initial 'maths' was likely wrong.

*What the dashboard said:          
                                
- Requests running: 20–35 concurrent ✓            
- Queue depth: 0 — no admission backpressure
- KV cache: 0% — vLLM has huge unused headroom    
- vLLM E2E P95: ~2 s — vLLM itself is healthy at the call level                                    
- TTFT P95: ~200 ms — prefill cheap               
- Prefix cache hit rate: 80-90%                                     
- Preemptions: 0  

So with vLLM healthy (kv @ 0%, queue @ 0, no preemptions), but per-call P95 so large at this concurrency I changed 'max-num-batched-tokens' parameter from 2048->4096 in order to reduce competition for the per step budget so the prefil stalls fewer decode steps. I then ran the same profile again.

│            │  Before  │  After  │    Δ     │    
  ├────────────┼──────────┼─────────┼──────────┤ 
  │ Achieved   │ 8.33     │ 8.33    │ —        │  
  │ RPS        │          │         │          │ 
  ├────────────┼──────────┼─────────┼──────────┤  
  │ OK / 500 / │ 2613 /   │ 2614 /  │ flat     │    
  │  timeout   │ 383 / 3  │ 384 / 1 │          │ 
  ├────────────┼──────────┼─────────┼──────────┤    
  │ P50        │ 1.75 s   │ 1.68 s  │ −4%      │    
  ├────────────┼──────────┼─────────┼──────────┤  
  │ P95        │ 10.66 s  │ 10.42 s │ −2%      │    
  │            │          │         │ (noise)  │  
  ├────────────┼──────────┼─────────┼──────────┤
  │ P99        │ 16.4 s   │ 17.7 s  │ +8%      │
  │            │          │         │ worse    │    
  ├────────────┼──────────┼─────────┼──────────┤
  │ Max        │ 54.8 s   │ 40.9 s  │ −25%     │    
  └────────────┴──────────┴─────────┴──────────┘
                                                  
  See screenshots/grafana_after.png. The targeted   
  vLLM metric (per-call latency) barely moved; P99
  regressed slightly; P95 stayed firmly past SLO - the   approach was wrong.                        
                                         
  KV at 0% and queue at 0 already told us vLLM had  
  headroom. The bottleneck is agent-side 
  serialisation: 3 sequential vLLM calls * up to 3  
  revise iterations = up to 9 round-trips per user
  request.      
                  
  I think overall, the diagnosis was right, the     
  lever was wrong. Real movers given more time would
   be (1) reducing MAX_ITERATIONS 3→2 (cuts         
  worst-case fan-out 33%), (2) running multiple
  uvicorn workers on the agent (parallelise       
  sequential agent calls across requests), (3)
  fixing the schema-render FK crash that's eating 
  12.8% of throughput on retries.  
                                
  Quality survived                      
           
  Re-ran the eval against the tuned vLLM —          
  results/eval_after_tuning.json:        
                                                    
  ┌──────────────┬───────────┬─────────────────┐ 
  │              │ Baseline  │   After-tune    │ 
  ├──────────────┼───────────┼─────────────────┤ 
  │ Overall pass │ 30%       │ 30%             │ 
  │  rate        │           │                 │ 
  ├──────────────┼───────────┼─────────────────┤ 
  │ Per-iter     │ 30 / 27 / │ 30 / 30 / 30    │    
  │ 1/2/3        │  30       │                 │ 
  ├──────────────┼───────────┼─────────────────┤    
  │ Per-DB rates │ (9 DBs)   │ bit-for-bit     │
  │              │           │ identical       │    
  └──────────────┴───────────┴─────────────────┘
                                                    
  Expected — vLLM throughput tuning doesn't touch
  agent correctness.            
                                         
  Verdict         
                                                  
  SLO missed. P95 = 10.42 s vs 5 s target — ~2x over
   budget, ~12% error rate on top. Phase 1 config 
  was already over-provisioned for vLLM (KV cache 0%
   under load); the gap is structural to the agent's
   sequential call graph, not the inference layer.
                                      

## Phase 7 - Wrap up

**Did the agent loop earn its keep?** Not at this sample size. Per-iteration pass rate was 30 / 27 / 30 — flat. The iter 2 dip is within noise (one question is 3.3% on a sample of 30). 8 of the 30 questions ran all the way to iter 3 so the loop is working, but the wins and losses cancel out — the productive revise on codebase_community Q28 vs. the harmful revise on card_games Q15 in Phase 3 are representative. Root cause is structural: without BIRD's `evidence` field in the prompt (we drop it per the task author's guidance), verify has no information generate didn't already have. It can tell that an answer is wrong (empty rows, schema mismatch) but can't reliably say *what's* wrong. Another participant reported 33% -> 53% with evidence surfaced, so the loop probably tips positive once you let it use the hints.

**What I'd do with more time** — in rough order of expected impact:

1. **More uvicorn workers on the agent server.** Phase 6 showed vLLM was sitting on huge headroom (KV cache 0%, queue 0) while the agent was bottlenecked on its own 3-call-per-request serialisation. Adding workers would parallelise across requests — this is the actually-binding lever I missed during the GPU window.
2. **`MAX_ITERATIONS 3 -> 2`.** Cuts worst-case fan-out by a third. Costs ~3% expected pass rate (the productive-revise rate) and gives back tail latency in roughly the same proportion. Cheapest possible iteration.
3. **Fix the FK rendering crash in `agent/schema.py`** that's behind the 12.8% HTTP 500 rate. Deterministic per DB so an easy patch — I left it alone per the assignment author's guidance but it's eating ~12% of throughput on retries.
4. **Guided decoding for the verify JSON output.** vLLM has a `--guided-decoding-backend` that would replace my defensive parser and its silent `ok=true` fallback on parse failures — verify becomes structurally honest.
5. **fp8 vs bf16 quality A/B** on the same 30 questions. I picked fp8 from first principles in Phase 1 but never actually measured the accuracy cost on this workload.
6. **Per-DB latency breakdown in Grafana.** The dashboard currently lumps all questions together but the long-tail DBs (toxicology, formula_1) dominate P99 and that should be visible.
