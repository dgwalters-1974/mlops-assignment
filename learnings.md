# Learnings — assignment study notes

Personal reference notes built up while working through the text-to-SQL
assignment. Sections 1–10 cover Phase 1 (vLLM serving config). Section 11+
covers Phase 3 (agent design and implementation).

---

## Contents

1. [Vocabulary](#1-vocabulary)
2. [The observability stack — what each tool does](#2-the-observability-stack)
3. [SLO decomposition — turning a target into per-call budgets](#3-slo-decomposition)
4. [Little's Law — the concurrency target](#4-littles-law)
5. [Model facts — Qwen3-30B-A3B](#5-model-facts)
6. [VRAM budget on an H100 80GB](#6-vram-budget)
7. [MoE — compute vs memory](#7-moe-compute-vs-memory)
8. [Phase 1 constraint summary](#8-phase-1-constraint-summary)
9. [Round 2 — Lever discovery](#9-round-2--lever-discovery)
    - [9.1 `--max-model-len` and the two-bound calculation](#91---max-model-len-and-the-two-bound-calculation)
    - [9.2 `--quantization fp8` and why H100 makes this free](#92---quantization-fp8-and-why-h100-makes-this-free)
    - [9.3 `--max-num-seqs` and the "two limits, lowest wins" pattern](#93---max-num-seqs-and-the-two-limits-lowest-wins-pattern)
    - [9.4 `--kv-cache-dtype fp8` and the preemption failure mode](#94---kv-cache-dtype-fp8-and-the-preemption-failure-mode)
    - [9.5 `--enable-prefix-caching` — the agent-workload superpower](#95---enable-prefix-caching--the-agent-workload-superpower)
    - [9.6 `--enable-chunked-prefill` + `--max-num-batched-tokens` — the prefill/decode tradeoff](#96---enable-chunked-prefill--max-num-batched-tokens--the-prefilldecode-tradeoff)
    - [9.7 `--gpu-memory-utilization` and asymmetric failure modes](#97---gpu-memory-utilization-and-asymmetric-failure-modes)
10. [Phase 1 config — plain-language quick reference](#10-phase-1-config--plain-language-quick-reference)
11. [Phase 3 — Agent design (Round 1)](#11-phase-3--agent-design-round-1)
    - [11.1 The graph shape and the node pattern](#111-the-graph-shape-and-the-node-pattern)
    - [11.2 Failure taxonomy for the verifier](#112-failure-taxonomy-for-the-verifier)
    - [11.3 Loop dynamics and termination](#113-loop-dynamics-and-termination)
    - [11.4 What revise needs to see (state field selection)](#114-what-revise-needs-to-see-state-field-selection)
    - [11.5 Round 1 design summary](#115-round-1-design-summary)
12. [Phase 3 — Prompt writing (Round 2)](#12-phase-3--prompt-writing-round-2)
    - [12.1 Generate SQL prompts](#121-generate-sql-prompts)
    - [12.2 Verify prompts](#122-verify-prompts)
    - [12.3 Revise prompts and the shared-system-prompt trick](#123-revise-prompts-and-the-shared-system-prompt-trick)
    - [12.4 Round 2 cross-cutting principles](#124-round-2-cross-cutting-principles)

---

## 1. Vocabulary

| Term | Meaning |
|---|---|
| **o11y** | "observability" — numeronym (11 letters between o and y). Same pattern: `i18n`, `k8s`, `a11y`. |
| **Observability** | The ability to understand what a running system is doing from the data it emits, without redeploying or adding print statements. |
| **SLI** (Service Level Indicator) | A metric on its own. e.g. "P95 latency." |
| **SLO** (Service Level Objective) | A *target* for an SLI, with a threshold and a window. e.g. "P95 latency < 5s over a 5-minute window." |
| **SLA** (Service Level Agreement) | A *contractual* promise to a customer, usually with money attached if you miss. Always looser than the internal SLO. |
| **P95 latency** | Sort all request latencies fastest → slowest; the value at the 95% mark is the P95. 95% of requests are at or below this number; 5% are slower. Captures the tail in a way averages don't. |
| **RPS** | Requests per second. The throughput half of most SLOs. |
| **Concurrency** | Number of requests in flight at the same time. Distinct from RPS — see Little's Law. |

---

## 2. The observability stack

This project uses four pieces that together cover three perspectives.
The first three are *live observability*; the fourth is *offline measurement*.

| Tool | Granularity | What it answers |
|---|---|---|
| **Prometheus** | Aggregate, live | Stores time-series numbers scraped from `/metrics` endpoints. You rarely look at it directly. |
| **Grafana** | Aggregate, live | Dashboards over Prometheus. "Is the serving layer healthy right now? Where is it bottlenecking?" |
| **Langfuse** | Per-request, live | Captures one trace per agent run, with each LLM call as a nested span. "Why was *this specific* request slow?" |
| **Evals** | Aggregate, offline | A Python script that runs a fixed test set through the agent and scores correctness. "Does the system get the right answer?" — independent of speed. |

### How they complement each other

- **Grafana** tells you *something* is slow.
- **Langfuse** lets you click into one slow request to see *which step* and *what the prompt was*.
- **Evals** confirm whether a tuning change broke quality.

Phase 6 forces you to use all three together: change a flag → Grafana confirms the targeted metric moved → Langfuse confirms no step regressed → evals confirm correctness held up.

### Where each tool sits in this repo

- Prometheus config: `infra/prometheus.yml` (scrapes vLLM on host:8000)
- Grafana dashboards: `infra/grafana/provisioning/dashboards/` (starter at `serving.json`)
- Langfuse wiring: `agent/server.py` (callback handler initialized from env keys)
- Evals: `evals/run_eval.py` + `evals/eval_set.jsonl` (30 questions)

---

## 3. SLO decomposition

The given SLO:

> *P95 end-to-end agent latency under 5 seconds, 10+ RPS over a 5-minute window.*

That's two SLOs stapled together — a latency SLO and a throughput SLO. You have to hit both.

### Per-call latency budget

The agent makes ~3 sequential LLM calls per request (generate_sql → verify → maybe revise). Because they're sequential, their latencies sum.

```
Per-call LLM budget  =  total budget / number of sequential LLM calls
                     ≈  5s / 3
                     ≈  1.67s per call (naive)
                     ≈  1.5s per call (with a small buffer for sqlite/Python overhead)
```

**Why this matters:** when staring at Langfuse traces in Phase 6, any single LLM call exceeding ~1.5s is a suspect for blowing the 5s end-to-end budget. The budget converts a vague "system is slow" into specific, accountable per-step limits.

### The general habit

When you have an end-to-end target, **decompose it into per-step budgets *before* measuring anything**. Then when real numbers arrive you can immediately point to "this step is over budget" — without the budget you'd just see numbers with no way to know which were problems.

---

## 4. Little's Law

> **At steady state: items in flight = arrival rate × average time in system.**
>
> **L = λ × W**

### Grocery store intuition

- Two customers arrive per minute (λ).
- Each customer takes 3 minutes at the register (W).
- At any moment, six customers are in flight in that aisle (L = 2 × 3).

### Why "10 RPS at 5s latency" isn't a contradiction

Naive intuition says: if each request takes 5s, how can the system handle 10 per second?
Answer: **requests overlap.** vLLM (and any modern serving system) processes many requests in parallel — batched inference on the GPU. The 5s latency and 10 RPS coexist because **50 requests are in flight at any moment**.

### The concurrency target for this project

```
L  =  10 RPS  ×  5s  =  50 concurrent agent requests in flight
```

Each in-flight agent is almost always inside an LLM call (sqlite + Python overhead is negligible). Therefore vLLM has to handle **~50 concurrent calls** at steady state.

**This is the central number for Phase 1.** Most config flags either enable or block hitting it.

---

## 5. Model facts — Qwen3-30B-A3B-Instruct-2507

| Property | Value | Why it matters for Phase 1 |
|---|---|---|
| Total parameters | 31B | The VRAM cost — *all* of these have to live in GPU memory. |
| Activated parameters per token | 3.3B | The compute cost. Decode is ~10× cheaper than a dense 31B. |
| Transformer layers | 48 | KV cache scales linearly with layers — each concurrent request stores KV across all 48. |
| Native max context | 262,144 (256K) | **Capability, not operating point.** If you don't cap context yourself, vLLM may pre-allocate KV slots assuming 256K-token requests — wildly wasteful for your real ~3K prompts. Capping `--max-model-len` is the single biggest free win in the config. |

Source: <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507>

---

## 6. VRAM budget

H100 has 80 GB HBM. The serving system's memory is a zero-sum game between weights, KV cache, and framework overhead.

> **Note:** the table below uses a simplified `80 GB` total budget for clarity. In reality vLLM reserves a safety margin via `--gpu-memory-utilization` (default 0.9). The realistic KV-pool figures are ~5 GB smaller; see [9.7](#97---gpu-memory-utilization-and-asymmetric-failure-modes) for the corrected math. The 3.4× ratio and qualitative argument below are unchanged.

| Item | bf16 weights | fp8 weights |
|---|---|---|
| GPU total (simplified) | 80 GB | 80 GB |
| Weights (31B × bytes-per-param) | −62 GB | −31 GB |
| Framework + activations + workspace | −5 GB | −5 GB |
| **KV cache headroom (simplified)** | **13 GB** | **44 GB** |
| KV pool corrected for default 0.9 utilization | ~6 GB | ~37 GB |
| KV pool corrected for chosen 0.92 utilization | ~8 GB | ~39 GB |

### The punchline

Quantizing weights to fp8 doesn't just save 31 GB on the weights — that 31 GB **falls into the KV cache budget**, giving a **3.4× larger KV pool**. Every byte you don't spend on weights is a byte you can spend on serving more concurrent requests.

Connecting back to Little's Law: you need ~50 concurrent requests. Per-request KV at a few thousand tokens of context is on the order of *a few hundred MB*. Therefore:

- **13 GB KV pool** → fits maybe a couple dozen concurrent requests. *Tight or insufficient* for the 50-target.
- **44 GB KV pool** → fits dozens to a hundred-plus concurrent requests. *Comfortable headroom.*

That is the complete, specific case for fp8 in this config. Not "fp8 is generally faster" — the precise rationale is *"my workload requires ~50 concurrent and bf16 doesn't give me the KV budget for that on this GPU."*

### Two separate fp8 knobs

1. `--quantization fp8` → **weights** stored as fp8. Saves the 31 GB shown above.
2. `--kv-cache-dtype fp8` → **KV cache entries** stored as fp8. Halves per-request KV cost.

They compose. Stacking both maximizes concurrency. Quality cost in both cases is small (~0.5–2 points on most benchmarks), and Phase 5 evals are designed to catch any regression.

---

## 7. MoE — compute vs memory

Qwen3-30B-A3B is a **Mixture-of-Experts** model. The "A3B" means ~3.3B *active* parameters per token, out of 31B total.

### Why compute is cheap

In an MoE layer, the dense feed-forward network is replaced by a **router + a pool of experts** (each expert is itself a small FFN). The router picks the top-k experts (often 2 or 8) for each token, and only those experts' matrices are multiplied. Attention layers stay dense.

**Compute per token scales with active params, not total.** That's the cheap-decode win.

### Why memory is *not* cheap

The router decides per-token, at inference time, which experts to call. Different tokens in the same batch route to different experts. There's no way to know in advance which experts can be skipped, so **all of them have to be resident in VRAM**.

**MoE saves FLOPs, not bytes.**

### Why this model fits a single H100

The assignment's choice of Qwen3-30B-A3B for a single-H100 setup isn't accidental — MoE threads a specific needle:

- **Decode is fast** (3.3B active) → helps the per-call latency budget.
- **Weights are large** (31B total in VRAM) → fights for KV cache space, hence fp8 matters.

A dense 30B would have the same VRAM problem with ~10× the decode cost (much harder to hit latency SLO). A dense 3B would have cheap decode but worse quality. Qwen3-30B-A3B gets dense-30B-ish quality at dense-3B-ish decode cost, paying with dense-30B memory pressure.

---

## 8. Phase 1 constraint summary

The quantitative spec that any candidate config must satisfy:

| Constraint | Value | Derivation |
|---|---|---|
| Per-call latency budget | ~1.5–1.7s | SLO ÷ 3 sequential LLM calls |
| Concurrent requests vLLM must handle | ~50 | Little's Law: 10 RPS × 5s |
| Total VRAM | 80 GB | hardware |
| Weights at bf16 / fp8 | 62 / 31 GB | 31B × bytes-per-param |
| KV cache pool at bf16 / fp8 | 13 / 44 GB | 80 − weights − 5 GB overhead |
| Native context | 256K (capability) | cap with `--max-model-len`; big lever |
| Compute cost per token | ~3.3B-active equivalent | MoE; decode is cheap |
| Layers | 48 | KV cache scales linearly with this |

These numbers are the rationales the Phase 1 `REPORT.md` entries should reference. Every flag picked in the launch script should trace back to one of these constraints.

---

## 9. Round 2 — Lever discovery

The Round 2 process: take a candidate bottleneck → check whether it threatens a Round 1 constraint → find the vLLM flag that controls it → pick a starting value with a rationale tied to a Round 1 number.

### 9.1 `--max-model-len` and the two-bound calculation

#### What the flag actually does

A common (and wrong) intuition: "vLLM pre-allocates KV cache for the full model context per request, so capping it saves memory linearly." That's not what happens.

vLLM uses **PagedAttention** — the KV cache pool is divided into small fixed blocks (default 16 tokens). When a request arrives, vLLM allocates blocks on demand as tokens are generated. A request that uses 2K tokens consumes ~125 blocks, not 16K. The KV pool is shared across many requests, each holding only what they've actually used.

So why does `--max-model-len` matter?

1. **Admission planning.** When vLLM decides whether to admit a new request into the running batch, it has to plan for the worst case — *"what if this request grows to max-model-len?"* With a 256K cap, it's conservative; with an 8K cap, it can confidently pack many more requests into the same pool.
2. **Bounds and CUDA-graph sizing.** It bounds input-length validation and sizes some internal scheduling structures.

**Practical effect:** a loose `--max-model-len` limits effective concurrency even when actual requests are small.

#### How the flag is picked — two bounds

The chosen value has to satisfy two constraints simultaneously.

**Lower bound — must cover the largest real request.**

For one LLM call inside the agent, the rough token budget:

| Component | Typical | Tail (worst case) |
|---|---|---|
| System prompt | 200–400 | 500 |
| DB schema | 500–2000 | 4000–5000 |
| User question | 50–150 | 200 |
| Prior SQL + exec result (revise call only) | 0 | 500–1000 |
| Output | 50–300 | 500 |
| **Total** | **~800–2800** | **~5500–7000** |

So **lower bound ≈ 7000 tokens** — below it and some tail BIRD question on a big-schema DB will fail with context-overflow.

**Upper bound — must let 50 concurrent fit in the KV pool.**

```
peak KV pool used  =  num_concurrent × max-model-len × per-token KV cost
                  ≤  KV pool size (Round 1 constraint #5)
```

Per-token KV size formula:

```
per-token per layer  =  2 (K and V) × num_kv_heads × head_dim × bytes_per_element
per-token total      =  per-token per layer × num_layers
```

For Qwen3-30B-A3B with assumed GQA params (verify in `config.json` for exact numbers):

- 48 layers, ~4 KV heads, head_dim ~128
- bf16 KV: `2 × 4 × 128 × 2 × 48` ≈ **96 KB per token**
- fp8 KV: ≈ **48 KB per token**

Worked example at `max-model-len = 8192`:

```
peak per request   =  8192 × 96 KB     ≈  790 MB (bf16 KV)
                                       ≈  395 MB (fp8 KV)

peak for 50 concurrent  =  50 × 790 MB  ≈  39 GB (bf16 KV)
                        =  50 × 395 MB  ≈  20 GB (fp8 KV)
```

Compared to the KV pool (pool figures here use the simplified-80-GB budget; [9.7](#97---gpu-memory-utilization-and-asymmetric-failure-modes) gives the corrected numbers at our chosen 0.92 utilization in the rightmost column):

| Config | KV pool (simplified) | KV pool (corrected @ 0.92) | Peak demand at max-model-len=8192 | Fits? |
|---|---|---|---|---|
| bf16 weights, bf16 KV | 13 GB | ~8 GB | 39 GB | ❌ way over |
| fp8 weights, bf16 KV | 44 GB | ~39 GB | 39 GB | ⚠️ at the corrected number, *exactly* at the limit — zero slack |
| fp8 weights, fp8 KV | 44 GB | ~39 GB | 20 GB | ✅ comfortable (~19 GB slack) |

**Important consequence of the correction:** without `--kv-cache-dtype fp8`, the realistic numbers say peak load uses *all* the KV pool. Any burst → preemption. fp8 KV moves from "nice to have" to "effectively required for stability." See 9.4 for the preemption discussion.

For `max-model-len = 16384`: peak demand doubles to ~79 GB (bf16 KV) / ~39 GB (fp8 KV). Doesn't fit comfortably with 50 concurrent. **16K is too loose.**

**So upper bound ≈ 9000 tokens** (with fp8 weights, bf16 KV) or **~18000 tokens** (with fp8 weights + fp8 KV stacked).

#### Why 8192 specifically

| Threshold | What it satisfies |
|---|---|
| `> 7000` | Covers tail real requests (lower bound) |
| `≤ ~9000` (fp8 weights only) | Lets 50 concurrent fit in KV pool (upper bound) |
| `8192` | Smallest power of 2 inside that band |

Powers of 2 are conventional because vLLM and CUDA prefer block-aligned sizes for graph compilation. 7168 or 8000 would also work; 8192 is the standard idiom.

With `--kv-cache-dtype fp8` added, the upper bound jumps to ~18K, so 16384 becomes defensible if you want more tail headroom. 8192 is the more conservative pick.

#### Hard inputs vs soft inputs

- **Hard** (just arithmetic): `peak_KV = num_requests × max_model_len × per_token_KV`.
- **Hard** (verified from `config.json`): per-token KV size is `2 (K and V) × num_key_value_heads × head_dim × bytes × num_hidden_layers` = `2 × 4 × 128 × 2 × 48` = **98,304 bytes ≈ 96 KB per token at bf16** (48 KB at fp8). The "~96 KB" estimate used throughout this document is the exact value, confirmed against Qwen3-30B-A3B-Instruct-2507's `config.json` (`num_key_value_heads=4`, `head_dim=128`, `num_hidden_layers=48`).
- **Soft** (estimate): "tail prompts hit 5–7K." To make hard: render the schema for every BIRD DB you have and measure prompt lengths empirically. Worth doing in Phase 6 if `--max-model-len` becomes a suspect.

#### Decision

```
--max-model-len 8192
```

**One-liner rationale for `REPORT.md`:**

> *Real prompts are ≤3K tokens with short outputs (tail ~5–7K); capping at 8K (vs. the model's 256K native) cuts vLLM's worst-case-per-request by ~32×, which is what makes the 50-concurrent target (Little's Law on 10 RPS × 5s) fit in the 44 GB fp8-weights KV pool.*

#### Recurring habit from this example

When picking any cap-style flag (`--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`), always find both bounds before picking a value:

1. **Lower bound** = the smallest value that doesn't break real workloads.
2. **Upper bound** = the largest value that doesn't break a constraint (memory, concurrency, latency).

Then pick inside the band — usually toward the lower end for safety, snapped to a power of 2.

### 9.2 `--quantization fp8` and why H100 makes this free

#### What's at stake

Round 1 constraint #5 (KV pool size) directly determines whether 50 concurrent fits. At bf16 weights the pool is 13 GB; the math from 9.1 shows 50 concurrent at 8K context demands ~39 GB of peak KV. **bf16 makes the SLO infeasible** on this hardware — this isn't a marginal optimization, it's a precondition.

#### What the flag does mechanically

- At load time, each weight tensor is rescaled by a per-tensor (or per-channel) scale factor and cast from bf16 → fp8.
- The scale is stored alongside the fp8 values and reapplied during compute.
- Matmuls run in fp8 on H100's native fp8 tensor cores; the accumulator stays wider (typically fp32) to avoid precision loss, then casts back down.
- Each weight takes 1 byte instead of 2 → 31 GB freed from the weight budget falls straight into the KV cache pool.

The exact format is usually written `fp8_e4m3` (4 exponent bits, 3 mantissa) — the inference standard. There's also `fp8_e5m2` (more dynamic range, less precision) for training. vLLM defaults are sensible; usually no need to override.

#### Why fp8 is essentially free on H100 (and *wasn't* on A100)

| GPU | FP8 tensor cores | Effect |
|---|---|---|
| A100 (Ampere, 2020) | ❌ none | Memory savings only; compute may be slower (emulated). |
| **H100** (Hopper, 2022) | ✅ native | Memory savings **and** ~2× matmul speedup vs bf16. |

On H100, fp8 wins on two axes simultaneously — memory *and* compute. This is the reason the assignment specifies H100; the same config on A100 would buy you concurrency but not speed.

#### Quality tradeoff

Quantization throws away precision; some of that loss can leak into output quality.

- **Standard benchmarks (MMLU, GSM8K, HumanEval):** ~0.5–2 point drop, usually noise-level.
- **SQL generation specifically:** usually fine — highly structured task with strong learned patterns. The verify step in the agent acts as a backstop for borderline outputs.
- **MoE-specific caveat:** mixture-of-experts models *can* be more sensitive to quantization than dense ones, because reduced-precision routing decisions can shift which experts fire. Qwen3-30B-A3B handles vLLM's standard fp8 path well in practice.
- **Where you'd detect a real problem:** Phase 5 execution-accuracy evals. That's the whole point of having an offline quality signal independent of latency — to catch regressions caused by performance tuning.

#### On-the-fly vs pre-quantized variants

Two paths:

1. **`--quantization fp8`** — vLLM quantizes the bf16 weights at load time using simple per-tensor scales. Fast to set up, no extra download.
2. **Load a pre-quantized HF variant** (e.g., from `RedHatAI` or `neuralmagic`) — quantized offline with calibration data (representative inputs used to compute better scales per channel). Usually slightly higher quality than on-the-fly.

Default to on-the-fly for Phase 1; swap to a calibrated variant only if Phase 5 evals show a quality gap.

#### Decision

```
--quantization fp8
```

**One-liner rationale for `REPORT.md`:**

> *bf16 weights leave only 13 GB for KV cache, which makes the 50-concurrent target (Little's Law on 10 RPS × 5s) infeasible. fp8 weights free 31 GB into the KV pool (→ 44 GB) and run at ~2× speed on H100's native fp8 tensor cores. Quality risk monitored via Phase 5 execution-accuracy evals.*

#### Recurring habit from this example

For any precision/quantization flag, **name the detection mechanism** for the quality cost — not just *"I accept some quality loss,"* but *"I accept it, and X will tell me if it's too much."* Without the detection mechanism, you can't tell a good tradeoff from a silent regression.

### 9.3 `--max-num-seqs` and the "two limits, lowest wins" pattern

#### What's at stake

Round 1 constraint #2 needs ~50 concurrent calls. vLLM has an explicit cap (separate from the KV pool) on how many sequences can be in the running batch at once. **If this cap is set below 50, the SLO is impossible regardless of how empty the KV pool is.**

#### What the flag does mechanically

vLLM's scheduler maintains a *running batch* — the set of sequences being processed (prefill or decode) in each forward pass. `--max-num-seqs` hard-caps how many distinct sequences can be in that batch simultaneously.

**Default:** typically 256 (varies by vLLM version; verify with `vllm serve --help`).

#### The "two limits, lowest wins" rule

> *vLLM has two concurrency limits — `--max-num-seqs` and what KV cache can physically hold. The lower one wins.*

| Scenario | Outcome |
|---|---|
| max-num-seqs too low | KV pool sits half-empty; requests queue unnecessarily. *Wasted hardware.* |
| max-num-seqs too high | KV becomes the binding constraint earlier than the scheduler expected; extra slots sit unused. *Wasted scheduler overhead.* |
| max-num-seqs ≈ KV-derived ceiling | Both limits bind at roughly the same load. *Right-sized.* |

#### Two-bound math (habit #7 in action)

**Lower bound:**

```
target concurrency   = 50           (Little's Law, constraint #2)
burst headroom       = ~20–30%      (real RPS isn't perfectly smooth)
lower bound          ≈ 60–65
```

**Upper bound** (from 9.1's KV-pool table at max-model-len=8192; corrected values use the realistic 39 GB pool from 9.7):

| KV config | Max concurrent (simplified pool) | Max concurrent (corrected @ 0.92 utilization) |
|---|---|---|
| fp8 weights, bf16 KV | ~55 | ~50 |
| fp8 weights, fp8 KV (covered in 9.4) | ~110 | ~100 |

#### Flag interaction

Bottleneck #3's right value depends on Bottleneck #4 (`--kv-cache-dtype`). Without fp8 KV, the corrected KV ceiling is ~50 — *exactly* the target — so `--max-num-seqs` set much above that just queues on KV. With fp8 KV, the ceiling rises to ~100 and the cap can be set looser.

#### Decision

```
--max-num-seqs 64
```

- Comfortably above the 50 target.
- Power of 2 (CUDA/vLLM prefer block-aligned sizes).
- Close to the no-fp8-KV ceiling (~55), so works even if fp8 KV is disabled for quality reasons.
- Room to bump to 96–128 in Phase 6 if load tests show queuing on this cap.

**One-liner rationale for `REPORT.md`:**

> *Little's Law puts steady-state concurrency at ~50 in-flight requests; setting max-num-seqs to 64 gives ~28% burst headroom while staying close to the KV-pool-derived ceiling (~55 at fp8 weights + bf16 KV, ~110 with fp8 KV), so neither the scheduler cap nor the KV pool becomes a wastefully premature bottleneck.*

#### Recurring habit from this example

**Don't tune interacting flags one at a time.** `--max-num-seqs` only makes sense after you've decided on `--kv-cache-dtype`. Choose flags as a *coherent set* and then revise individual values within it.

### 9.4 `--kv-cache-dtype fp8` and the preemption failure mode

#### What's at stake — feasibility vs stability

9.2 (fp8 weights) made the SLO *achievable*. But "achievable" is tight. With max-model-len=8192 and 50 concurrent, the KV pool peak is ~39 GB out of the (simplified) 44 GB pool — only ~5 GB of slack.

| Config (max-model-len=8192) | Peak at 50 concurrent | Pool (simplified) | Slack (simplified) | Pool (corrected @ 0.92) | Slack (corrected) |
|---|---|---|---|---|---|
| fp8 weights, bf16 KV | ~39 GB | 44 GB | **~5 GB** | ~39 GB | **~0 GB** |
| fp8 weights, fp8 KV | ~20 GB | 44 GB | **~24 GB** | ~39 GB | **~19 GB** |

The corrected pool from [9.7](#97---gpu-memory-utilization-and-asymmetric-failure-modes) makes the case stronger: without fp8 KV, the realistic slack at peak is **effectively zero** — any burst triggers preemption. fp8 KV moves from "nice to have" to "structurally required."

#### What preemption is (and why it shows up in tail latency)

When the KV pool fills up and a new request needs space, vLLM can **evict** a currently-running request:

1. The evicted request's KV blocks are freed.
2. The evicted request goes back to the waiting queue.
3. When space opens up, it gets re-admitted — but its KV cache is gone, so it must **re-prefill from scratch** (or recover partially via prefix cache, if enabled).

Cost: a request mid-generation is thrown back to "re-prefill everything." Effect: P99 latency spikes while P50 looks fine. In Phase 6, this is a pattern to look for in Grafana — `P99 climbing >> P50` is preemption's signature.

So this bottleneck doesn't threaten *feasibility* of the SLO; it threatens **stability**.

#### What the flag does mechanically

The K and V tensors in the cache are stored as fp8 instead of bf16. When attention reads them during a forward pass:

1. fp8 K/V values are cast up to compute precision (bf16/fp16) on the fly.
2. Attention math runs at the higher precision.
3. Storage halves: per-token KV 96 KB → 48 KB.

Everything downstream — pool capacity, concurrency ceiling, preemption likelihood — follows from that one halving.

#### Quality tradeoff (different from weight fp8)

KV quantization is **more sensitive** than weight quantization, because:

| | Weight fp8 | KV fp8 |
|---|---|---|
| When quantization error occurs | One-shot at load | Every token, every attention step |
| Error propagation | Localized to matmul output | Accumulates through the sequence |
| Typical eval drop | ~0.5–2 pts | ~1–3 pts |
| Long-context sensitivity | Low | Higher |

For this workload (short structured outputs from ~3K-token prompts), KV fp8 is a low-risk pick. **Detection via Phase 5 evals** (habit #8).

#### Why this is almost always worth turning on

| Axis | Effect |
|---|---|
| KV pool effective capacity | **2×** |
| Concurrency ceiling at 8K | 55 → 110 |
| Slack at 50 concurrent | 5 GB → 24 GB |
| Preemption likelihood | Much lower |
| Compute speed | ~unchanged |
| Quality drop | ~1–3 pts (detectable) |

The cost is small and detectable. The benefit is structural. On H100 with this workload, the question is *"what would have to be true for me **not** to turn it on?"* — and the only good answer is "Phase 5 evals show a real regression."

#### Decision

```
--kv-cache-dtype fp8
```

**One-liner rationale for `REPORT.md`:**

> *fp8 weights alone leave only ~5 GB of slack in the KV pool at peak 50 concurrent — one burst can trigger preemption, which spikes P99 latency. fp8 KV halves per-token KV (~96 KB → ~48 KB), restoring ~24 GB of slack and raising the concurrency ceiling to ~110. Attention is more quantization-sensitive than FFN, so the quality risk is slightly higher than weight-only fp8 — monitored via Phase 5 evals.*

#### Recurring habits from this example

- **Feasibility ≠ stability.** A config that meets SLO at average load can fail it under bursts. The size of headroom in the binding resource is what makes performance predictable.
- **Preemption is the failure mode of KV pressure.** In Phase 6, P99 ballooning while P50 stays steady is preemption's signature — trace it back to KV pool utilization.
- **Not all quantization is equal.** Weights are safest; activations are riskiest; KV sits between. Plan quality monitoring accordingly.

### 9.5 `--enable-prefix-caching` — the agent-workload superpower

#### Why this flag is so much bigger than it looks

Most vLLM flags trade memory for concurrency or quality for speed. Prefix caching is the rare one with **zero quality cost** and a 10–20× win on prefill latency for repeated prefixes. The reason it matters so much here is the *shape* of agent traffic.

#### The request structure that makes prefix caching transformative

```
Call #1: generate_sql
  system prompt    (~300 tokens)    ← SHARED with calls 2 & 3
  schema           (~1500 tokens)   ← SHARED with calls 2 & 3
  user question    (~100 tokens)    ← SHARED with calls 2 & 3
  → generates SQL  (~150 tokens)

Call #2: verify
  [identical prefix above]
  SQL from call 1  (~150 tokens)    new
  exec result      (~100 tokens)    new
  → verify verdict (~50 tokens)
```

Of the ~2000 tokens going into call 2, ~1900 are tokens the model has already seen in call 1. Without prefix caching, call 2 re-prefills all 2000. With it, only ~100 new tokens prefill — calls 2 and 3 become ~10–20× cheaper.

And **across requests on the same DB**, the schema (~1500 tokens) is identical, so the cache reuse extends beyond a single agent run.

#### Latency budget impact

```
Without prefix caching:  call 1 prefill 1.0s + call 2 prefill 1.0s + call 3 prefill 1.0s ≈ 3.0s
With prefix caching:     call 1 prefill 1.0s + call 2 prefill 0.05s + call 3 prefill 0.05s ≈ 1.1s
```

(Absolute numbers illustrative; ratios are the point.) Saving ~2 s out of the 5 s SLO is enormous — it's the single biggest one-flag improvement to P95 you can make.

#### What the flag does mechanically

1. vLLM organizes KV cache in fixed-size blocks (default 16 tokens, via `--block-size`).
2. Each block produced during prefill is **content-addressed** by a hash of (token IDs in block + parent block hash). The hash encodes the full prefix that produced this block's KV.
3. Cached blocks are added to a global pool, keyed by their hash.
4. When a new request arrives, vLLM hashes its prefix tokens in block-of-16 chunks from position 0 and looks each one up.
5. Matched blocks are **borrowed** — the new request's logical block table points at the existing physical block. No recompute, no extra memory.
6. Prefill only runs for tokens *after* the last cached block.
7. Cached blocks are LRU-evicted under KV pressure.

#### The two requirements for a cache hit

Prefix caching is **exact match** on token IDs from position 0:

1. **Identical token sequence at the start.** One different token at position 0 → no cache hit at all. Timestamps, random IDs, request UUIDs in the system prompt destroy this.
2. **Identical tokenization.** Same tokenizer config and chat template. Schema text must render byte-for-byte identical across calls.

In this project:

- `agent/schema.py` uses `@lru_cache` on `render_schema()` (line 27) → schema rendering is deterministic ✅
- System prompts in `agent/prompts.py` should be static strings (no timestamps) — design choice to preserve when writing them in Phase 3
- Prompt order matters: **shared content first, per-call content last** maximizes the cacheable prefix

#### Tradeoffs

| Axis | Effect |
|---|---|
| Compute | Big win — repeated prefill skipped |
| KV pool memory | Cached blocks occupy space, but are shared across requests → counted once, not N times |
| Quality | **Zero** (identical to recomputing) |
| Hit rate | Workload-dependent. Agents: huge. Ad-hoc chat: marginal. One-shot batch: zero. |
| Complexity | None at the API surface — flag flip |

#### Decision

```
--enable-prefix-caching
```

(In recent vLLM versions this is default-on; verify with `vllm serve --help`. Even if default-on, call it out in `REPORT.md` because it's so material to the latency math.)

**One-liner rationale for `REPORT.md`:**

> *Within one agent request, verify and revise calls share a ~1900-token prefix with generate_sql. Across requests on the same DB, the schema (~1500 tokens) is identical. Prefix caching skips re-prefilling those shared tokens, cutting prefill latency on calls 2 and 3 by ~10–20×. Zero quality cost; the single biggest latency lever for this workload.*

#### A/B opportunity for Phase 6

`--no-enable-prefix-caching` turns it off. Running one load-test cycle without prefix caching and comparing P95 vs the default config gives you concrete numbers for the "how much did this flag save?" claim in the report. Strong evidence.

#### Recurring habits from this example

- **Optimization is workload-dependent.** Prefix caching is transformative for agents, marginal for chat, useless for one-shot batch. Always read your request shape — what's shared, what varies — before picking optimizations.
- **Cache hits require determinism.** Anything that varies the prefix tokens (timestamps, request IDs, non-deterministic rendering) silently kills cache effectiveness.
- **Prompt structure affects performance.** *"Shared content first, variable content last"* is a prompt-engineering rule about latency on cache-aware servers, not just about model behavior.

### 9.6 `--enable-chunked-prefill` + `--max-num-batched-tokens` — the prefill/decode tradeoff

#### The contention this flag addresses

Every vLLM forward pass processes a mix of two workloads on the same GPU compute:

| Work type | What it does | Shape |
|---|---|---|
| **Prefill** | Compute KV cache for a new request's prompt | Big batch — e.g., 3000 tokens × 48 layers in one pass |
| **Decode** | Generate the next token for a sequence already in flight | One token per active sequence per step |

A single batch step has a token budget; that budget gets allocated between prefill work and decode work.

At steady state in this project:

- ~50 sequences in flight, all decoding
- New requests arriving at ~30 RPS (10 agent RPS × 3 calls)
- Each new request brings ~2–3K tokens of prefill (calls 1 = full; calls 2–3 partial after prefix cache)

#### The failure mode without chunking — head-of-line blocking

Without chunked prefill, a 2000-token prefill consumes almost an entire forward pass:

```
Step N:    [prefill 2000 tokens from request X]   ← takes ~all the step
           50 decoding sequences:  waiting...      ← zero decode this step

Step N+1:  [decode 50 sequences]                   ← each gets 1 token
```

Each prefill stalls all 50 decoders for a step. With ~30 prefills per second arriving, decode throughput craters and **P99 latency spikes while P50 looks fine**.

This is **head-of-line blocking** — one expensive operation in the queue stops everyone behind it. It's the failure mode of any queue without preemption.

#### What chunked prefill does

Splits long prefills into chunks that fit alongside decode work in the same step:

```
Step N:    [chunk 1: 1024 tokens] + [decode 50 sequences]  ← both happen
Step N+1:  [chunk 2: 976 tokens]  + [decode 50 sequences]  ← both happen
Step N+2:  [decode 50 sequences]                           ← prefill done
```

Decodes never get fully starved. The prefill itself takes slightly longer (split across more steps), but **decode tail latency stabilizes**. In a latency-SLO world that's almost always the right trade.

#### How the two flags interact

| Flag | Role |
|---|---|
| `--enable-chunked-prefill` | On/off switch (default-on in recent vLLM) |
| `--max-num-batched-tokens` | Token budget per batch step. With chunked prefill on, also sets the max chunk size for prefill. |

The lever you actively tune is `--max-num-batched-tokens`.

#### Sizing `--max-num-batched-tokens`

**Lower bound:** must accommodate decodes plus some prefill progress.

```
50 concurrent × 1 decode token each   = 50 tokens minimum for decode
+ room for some prefill                = ?
```

If set near 50, you'd have decode-only steps with no prefill progress. Need meaningfully more than the decode floor.

**Upper bound:** if `--max-num-batched-tokens` ≥ `--max-model-len`, a single prefill can swallow the whole step — chunking becomes meaningless and head-of-line blocking returns.

**Shape of the tradeoff inside the band:**

| Value | Behavior |
|---|---|
| 1024 | Tiny chunks, smooth decode, prefill takes many extra steps |
| 2048 | Reasonable middle — 3K prompts chunk into 2 pieces |
| 4096 | Most 3K prompts prefill in a single step, decodes share the remaining budget |
| 8192 | = max-model-len → no chunking effectively happens |

#### Decision

```
--enable-chunked-prefill           (or rely on default-on)
--max-num-batched-tokens 2048
```

Conservative starting value — prioritizes decode smoothness over peak prefill throughput. Phase 6 can push to 4096 if load tests show prefill is the bottleneck.

**One-liner rationale for `REPORT.md`:**

> *At ~50 concurrent decoders with new ~2-3K-token prefills arriving every ~33ms, an unchunked prefill stalls all decodes for a full forward pass — head-of-line blocking that wrecks P99. Chunked prefill with max-num-batched-tokens=2048 splits long prefills into chunks that fit alongside decode work in the same step, trading slightly slower prefill for much more predictable decode latency. The lever that protects P95/P99 stability under load.*

#### Recurring habits from this example

- **Throughput and latency are different optimization targets.** Configurations that maximize throughput (big batches, no chunking) often have terrible tail latency. SLO-driven systems often give up some throughput for predictability.
- **Head-of-line blocking is the failure mode of any queue without preemption.** Whenever there's a "one expensive thing processed fully before the next starts" structure, expect tail latency problems. Look for chunking/interleaving as the standard fix.
- **Read metrics in pairs.** A change that improves P50 might hurt P99 (or vice versa). Diagnosis quality in Phase 6 depends on always looking at both at once.

### 9.7 `--gpu-memory-utilization` and asymmetric failure modes

#### What the flag controls

The fraction of total GPU memory vLLM may use for its entire memory pool — weights + KV cache + activation workspace + CUDA workspace.

```
GPU total memory  =  80 GB (H100)
vLLM pool budget  =  total × gpu-memory-utilization
                  =  80 × 0.9 (default)  =  72 GB
```

The remaining 10% (~8 GB) is left unused by vLLM as a safety margin for:

- CUDA context overhead outside vLLM's tracking
- Other processes on the GPU
- Transient activation spikes during peak batch sizes
- Framework state vLLM doesn't account for

#### Correction to the section 6 / 9.1 numbers

Section 6's table used the simplified `80 GB − weights − 5 GB overhead`, which implicitly assumed `gpu-memory-utilization 1.0`. With the realistic 0.9 default (or our chosen 0.92):

| Term | Section 6 simplified | At 0.92 utilization (realistic) |
|---|---|---|
| vLLM budget | 80 GB | 73.6 GB |
| Less weights (fp8) | 31 GB | 31 GB |
| Less workspace | 5 GB | ~4 GB |
| **KV pool** | **44 GB** | **~39 GB** |

So earlier sections' "44 GB KV pool" is optimistic by ~5 GB. The conclusions still hold qualitatively, but the slack figures tighten:

- 9.1's table at `max-model-len 8192`: fp8 weights + bf16 KV peak (39 GB) is now essentially equal to the corrected pool (39 GB) → **infeasible margin** without fp8 KV.
- 9.4's "5 GB slack" → effectively **0 GB slack** without fp8 KV.
- 9.4's "24 GB slack" with fp8 KV → ~19 GB.

This makes the argument for `--kv-cache-dtype fp8` stronger, not weaker — without it, the corrected math says the SLO has zero burst headroom in the KV pool.

#### Why the default is 0.9

The 10% margin is conservative because failure here is **binary, not graceful**:

1. **Activation spikes.** Under peak batch + max-model-len, intermediate activations grow larger than vLLM's pre-flight measurement may anticipate.
2. **CUDA workspace.** Some kernels (fused attention especially) allocate workspace memory on first use; not always captured in vLLM's accounting.
3. **Other GPU processes.** Even "dedicated" cloud GPUs have driver state and monitoring agents consuming VRAM.
4. **OOM kills the whole server.** Not just one request — the running batch and potentially the process.

#### Gains vs risks when pushing higher

Each percentage point of utilization is ~0.8 GB:

| Setting | Pool budget | Gain over 0.9 | Risk |
|---|---|---|---|
| 0.9 (default) | 72 GB | — | None |
| 0.92 | 73.6 GB | +1.6 GB | Tiny |
| 0.95 | 76 GB | +4 GB | Modest |
| 0.98 | 78.4 GB | +6.4 GB | High — one bad spike → OOM |

#### The asymmetric tradeoff

Unlike most flags where the cost of mis-setting is graceful (slightly worse latency, slight quality drop), `--gpu-memory-utilization` set too high has a **binary catastrophic cost**: the server crashes.

```
Win from +1 point:   +0.8 GB pool  →  ~1-2 more concurrent requests of KV (diminishing returns)
Cost of going too far:  OOM crash of the whole server (not graceful)
```

This is the rare flag where *"be conservative and tune up in Phase 6 with evidence"* is genuinely correct — not a hedge.

#### Decision

```
--gpu-memory-utilization 0.92
```

Modest bump above the 0.9 default. Defensible because:
- Dedicated H100 VM, no competing GPU tenants
- Workload runs tight on KV (~39 GB pool needed for 50 concurrent at 8K + fp8 KV)
- Gain (~1.6 GB → ~3-4 more concurrent requests' KV headroom) is meaningful but not aggressive
- Failure mode is binary — staying close to default protects against unexpected activation spikes

**One-liner rationale for `REPORT.md`:**

> *Dedicated H100, no competing GPU tenants; workload runs tight on KV (~37 GB pool needed for 50 concurrent at 8K + fp8 KV). Bumping utilization from 0.9 to 0.92 buys ~1.6 GB additional KV pool (~3-4 more concurrent requests' headroom) at low OOM risk. Conservative bump because the failure mode is binary — Phase 6 can push higher with evidence.*

#### Recurring habits from this example

- **Some flags have asymmetric failure modes.** Most degrade gracefully (slower, more queuing). A few — memory utilization, OOM-related caps, max-model-len set below real prompt size — fail binary-catastrophically (server crash, request rejection). Be measurably more conservative with these.
- **"Conservative" isn't a synonym for "default."** Picking 0.92 over 0.9 is still a *choice* that requires rationale. Picking the default *because it's the default* — without articulating why it fits this workload — doesn't satisfy the grading rubric. Defaults need justification too.

---

## Recurring habits to internalize

1. **Decompose end-to-end targets into per-step budgets** before measuring.
2. **Use Little's Law** to convert RPS + latency targets into concurrency requirements.
3. **Memory is zero-sum.** Every byte spent on weights is a byte not spent on KV cache.
4. **Capability ≠ operating point.** Just because the model supports 256K context doesn't mean you should let it.
5. **MoE saves FLOPs, not bytes.** Always budget memory for the full parameter count.
6. **Every flag needs a rationale that points to a workload number** — not "this is generally good," but "I need N concurrent and the current setting blocks that."
7. **For any cap-style flag, find both bounds before picking a value.** Lower bound = smallest value that doesn't break real workloads. Upper bound = largest value that doesn't break a constraint. Pick inside the band, snap to a power of 2.
8. **For any precision/quantization flag, name the detection mechanism for the quality cost.** Not just "I accept some quality loss," but "I accept it, and *this signal* will tell me if it's too much." Without a detection mechanism you can't tell a good tradeoff from a silent regression.
9. **Match the flag to the hardware.** fp8 is "free" on H100, not on A100. Always check whether the optimization the docs recommend assumes hardware features you actually have.
10. **Don't tune interacting flags one at a time.** Some flags (e.g., `--max-num-seqs` and `--kv-cache-dtype`) only make sense together — picking one in isolation can give a value that's locally optimal but globally wrong once the other flag is set.
11. **Find the binding constraint.** Systems usually have one bottleneck at a time; the others are slack. Aim to size flags so multiple constraints become binding *at the same load*, not so one dominates while others sit idle.
12. **Feasibility is not stability.** A config that meets SLO at average load can fail it under bursts. Size headroom — not just capacity — in the binding resource.
13. **Preemption is the failure mode of KV pressure.** Symptom: P99 spikes while P50 holds. In Phase 6, trace this pattern to KV pool utilization before tuning anything else.
14. **Not all quantization is equal.** Weights → safest; KV → riskier (errors propagate through attention); activations → riskiest. Plan quality monitoring proportional to the precision target.
15. **Optimization is workload-dependent.** Read your request shape (what's shared, what varies) before picking flags. The same optimization can be transformative for one workload and useless for another.
16. **Cache hits require determinism.** Anything that varies prefix tokens silently kills cache effectiveness. Audit prompts and renderers for stability.
17. **Prompt structure affects latency, not just quality.** *"Shared content first, variable content last"* maximizes the cacheable prefix on cache-aware servers.
18. **Throughput and latency are different targets.** A throughput-maximizing config (big batches, no chunking) often has terrible tail latency. SLO-driven systems usually trade some throughput for predictability.
19. **Head-of-line blocking is the failure mode of any queue without preemption.** Symptoms: P99 spikes while P50 holds. Standard fix: chunking / interleaving.
20. **Read metrics in pairs.** A change that improves P50 might hurt P99. Always look at both, not one.
21. **Some flags have asymmetric failure modes.** Most degrade gracefully; a few fail binary-catastrophically (OOM, request rejection). Be measurably more conservative with these.
22. **"Conservative" is not a synonym for "default."** Picking the default value is still a choice and still requires articulated rationale — defaults are not free of justification.
23. **Node returns a partial state dict, not a full one.** Be explicit about what each node changes; let the graph framework merge. This forces clear thinking and makes traces diff-able.
24. **Never trust the model's surface format.** Whatever the prompt asks for — SQL, JSON, a yes/no — the response may arrive fenced, prefixed with prose, or sloppily formatted. Parse defensively at every LLM boundary.
25. **For looping graphs, the iteration-cap check must be unconditional.** Even if the "natural" exit condition is never satisfied, the cap is what guarantees termination. A graph without a working cap can deadlock under bad inputs.
26. **Prompt content selection is tokens-budget management.** Excluding a field isn't censorship — it's noise reduction, prefix-cache preservation, and attention focus. Include what changes the answer; exclude what doesn't.
27. **Output-format contracts use "ONLY" deliberately.** All-caps in a rules list is read by capable models as a hard constraint signal. *"Output ONLY the SQL..."* outperforms *"Output the SQL..."* on format reliability.
28. **For structured outputs, show literal examples in the prompt.** Models copy what they see more reliably than they follow descriptions of schemas. Two examples (one for each output case) beat a prose schema description.
29. **Across-step prefix caching requires identical system prompts.** Put step-specific framing in the user message, not the system message. `REVISE_SYSTEM = GENERATE_SQL_SYSTEM` is a deliberate sharing pattern, not laziness.
30. **Always pair "prompt asks for X" with "code defensively parses non-X."** The prompt is a request; the parser is the safety net. Capable models follow format contracts most of the time, not all of the time.

---

## 12. Phase 3 — Prompt writing (Round 2)

The six prompt strings in `agent/prompts.py` filled in. The mechanical writing was straightforward; the *design choices* are what's worth recording.

### 12.1 Generate SQL prompts

The foundational pair — sets the format contract the verify and revise prompts inherit.

#### Structural choices

- **System message has three jobs:** establish the role, list rules, specify the output format contract.
- **User message goes: schema → question → imperative.** Schema first because it's the most stable (same across all calls per DB) and benefits most from prefix caching. Question second (same across calls within one agent run). Closing imperative ("Please write the SQLite query.") helps the model reorient after a ~1500-token schema dump.
- **Token budget for the cacheable header:** system (~250 tokens) + schema (~1500 tokens) + question (~100 tokens) = ~1850 tokens cached after the first call.

#### Specific clauses worth noting

| Clause | Why |
|---|---|
| *"a single SQL query"* | Without "single," capable models sometimes return alternatives or reasoning + SQL. |
| *"Use only tables and columns from the provided schema"* | Anti-hallucination guard against invented column names. |
| *"Quote identifiers with double quotes"* | `agent/schema.py:_q` double-quotes everything; models trained on MySQL corpora sometimes default to backticks. |
| *"Output ONLY the SQL query, wrapped in a `\`\`\`sql ... \`\`\`` fence"* | The format contract with `_extract_sql()` (`graph.py:74-81`). The fence isn't strictly required (extractor falls back to whole reply) but "ONLY" prevents prose explanations from polluting the SQL. |

#### Three rules I added on top

- *"Prefer simple queries over complex ones"* — guards against the model reaching for CTEs when a single SELECT works. Simpler queries also fail less and are easier for verify to inspect.
- *"Use LIMIT for top-N questions"* — common omission for "top 5", "highest 3" style questions.
- *"Use COUNT(*) for counts"* — anchors counting questions to the canonical form.

#### Deliberate non-inclusions

- **No few-shot examples (zero-shot v1).** Adds ~500–1000 cacheable tokens but doubles the system prompt size. Worth revisiting if Phase 5 evals are weak.
- **No "think step by step."** This is a generation task, not a reasoning one. CoT adds latency and decode tokens without quality wins on SQL.
- **No BIRD-bench "evidence" hint handling.** Our `AgentState` doesn't surface the hint field; if Phase 5 reveals big gaps tied to missing hints, add then.

### 12.2 Verify prompts

The hardest of the three because it asks for *judgment* (not template-filling) and *parseable JSON* (LLMs mangle this regularly).

#### Why this prompt is structurally different from generate/revise

Verify's output isn't free-form text — it's a structured `{ok, issue}` JSON object. Two implications:

1. **The system prompt has to specify the JSON shape concretely.** Showing the two valid outputs (one for ok, one for not-ok) is more reliable than describing the schema in prose. *"Always output JSON with a boolean 'ok' field and a string 'issue' field"* leaks; showing literal examples doesn't.
2. **The user message order changes.** Verify uses `question → sql → execution → imperative`. Schema is **not included** — the verifier doesn't need to syntax-check SQL against the schema (the executor already did that). Excluding the schema saves ~1500 tokens per call. (Habit #26 — tokens-budget management.)

#### Surfacing the failure taxonomy in the prompt

The Round 1 taxonomy (errored / zero rows / wrong columns / semantic mismatch) is in the prompt as a numbered list. Reasons:

- Without it, the verifier tends to be either too lenient (rubber-stamps anything that ran) or too strict (rejects legitimate empty results).
- Numbering them makes them retrievable — the model is more likely to consider each failure mode systematically.
- Semantic mismatch is #4 specifically because it's the hardest case and benefits most from being made explicit.

#### The zero-rows caveat

> *Zero rows is sometimes the correct answer (e.g., "list students born in 1850" — there genuinely may be none). Use judgment.*

This is a false-positive guard. Without it, verifiers over-eagerly reject empty results that are actually correct. BIRD has plenty of these.

#### JSON output reliability

The prompt asks for fenced JSON, but `verify_node` (when we implement it in Round 3) will still need to parse defensively for:

- Output wrapped in `\`\`\`json` (handled by extraction)
- Prose before or after the JSON (handled by extraction + try/except)
- Python `True`/`False` instead of JSON `true`/`false` (defensive parsing)
- Missing keys (default to `ok=False, issue="parse error"` on bad parses)

**Recurring habit:** ask the prompt to produce clean JSON; assume the code receives sloppy JSON.

### 12.3 Revise prompts and the shared-system-prompt trick

The interesting design choice: **`REVISE_SYSTEM = GENERATE_SQL_SYSTEM`** (literal reuse).

#### Why share the system prompt

Prefix caching matches on **byte-identical token sequences from position 0**. If `revise` had a *different* system prompt — even just one word different — the cache would invalidate from position 0 on every revise call, and the model would re-prefill the entire ~1850-token header.

By sharing the system prompt:

- The ~250-token system message is cached after the first generate_sql call.
- The schema + question portion of the user message (~1600 tokens) is also cached, because revise uses the same schema and question in the same positions.
- Only the per-iteration content (failing SQL + execution + verify_issue, ~400 tokens) requires fresh prefill.

That's ~1850 tokens of cache hit per revise call. With 2 revises per agent request (worst case), the savings compound.

#### The tradeoff this requires

The shared system prompt has to be general enough to work for both writing-from-scratch and fixing-up cases. *"You write SQLite SQL queries"* works for both; *"You convert English to SQL"* might feel slightly weird in a revise context where the input is already SQL. The framing trick is to put the "this is a fix-it call" context in the **user** message, not the system message.

#### The user message structure

```
Database schema:        ← cacheable (same as generate_sql)
{schema}

Question: {question}    ← cacheable (same as generate_sql)

A previous attempt was rejected. Here is what happened:    ← divergence point — new content starts here

Previous SQL:
```sql
{sql}                   ← new per iteration
```

Result of running it:
{execution}             ← new per iteration

Why it was rejected: {issue}    ← new per iteration

Please write a corrected SQLite query that fixes this issue while still answering the original question. Don't change parts of the SQL that weren't called out as wrong.    ← closing imperative
```

#### Two clauses in the closing imperative — why both

- **"...while still answering the original question."** Without this, models sometimes over-correct toward the verifier's complaint and drift from the original ask. E.g., if verify says "missing gender filter," the model might add the filter but drop other parts of the original question.
- **"Don't change parts of the SQL that weren't called out as wrong."** Discourages random rewrites — important in semantic-mismatch cases where only one part is wrong. Anchors the model to *minimal* corrections.

### 12.4 Round 2 cross-cutting principles

Three patterns that came up across all three prompt pairs:

#### A. Output-format contracts are written *as if* they're hard rules

Every prompt ends with an "Output ONLY..." rule. The capitalization on "ONLY" matters — capable models read it as a constraint rather than a suggestion. The parsers (`_extract_sql`, the JSON parser we'll write) are still defensive, but the prompt does most of the work.

#### B. Show, don't describe (for structured outputs)

For verify's JSON output, the prompt **shows** two literal examples of valid output:

```json
{"ok": true, "issue": ""}
{"ok": false, "issue": "..."}
```

This is more reliable than describing the schema in prose ("output a JSON object with a boolean ok field and a string issue field"). The model copies the format it sees.

#### C. Prefix-cache-friendly order is non-negotiable

In every prompt, the order is:
```
1. System message (most stable)
2. Schema, if used (per-DB stable)
3. Question (per-agent-run stable)
4. Per-iteration content (last)
5. Closing imperative
```

Deviating from this order (e.g., putting the question first because it "feels more natural") sacrifices the prefix-cache wins Phase 1 worked so hard for. Habit #17 from Phase 1 is the same habit, applied here.

#### D. Same role, different user context > different roles

The `REVISE_SYSTEM = GENERATE_SQL_SYSTEM` decision generalizes: when two agent steps have similar output requirements, make their *system* prompts identical and put the per-step framing in the *user* message. This pattern is also how multi-turn chat systems achieve prefix caching across turns.

#### New habits this round added

- **Output-format contracts use "ONLY" deliberately.** Capable models treat the all-caps as a constraint signal.
- **For structured outputs, show valid examples in the prompt.** Models copy what they see more reliably than they follow descriptions of schemas.
- **Across-step prefix caching requires identical system prompts.** Put step-specific framing in the user message, not the system message.
- **Always pair "prompt asks for X" with "code defensively parses non-X."** The prompt is a request; the parser is the safety net.

---

## 10. Phase 1 config — plain-language quick reference

The final config from Round 3, with one-sentence plain-English explanations of why each flag is there. For the full rationale chains, see the corresponding section in 9.x.

| Flag | Value | Plain-language reason | Detail |
|---|---|---|---|
| `--quantization fp8` | on | Shrinks the model in memory so there's room to handle 50 requests at once. The H100 chip is built to run fp8 fast, so it's also a speed win. | [9.2](#92---quantization-fp8-and-why-h100-makes-this-free) |
| `--kv-cache-dtype fp8` | on | Halves the memory each request needs for its "scratchpad." Without it we'd be operating right at the edge — one bad burst would crash performance. | [9.4](#94---kv-cache-dtype-fp8-and-the-preemption-failure-mode) |
| `--max-model-len` | 8192 | Tells vLLM "no request will use more than 8K tokens." This stops vLLM from reserving room for the model's massive 256K native context (which we'll never use), letting it pack more requests in. | [9.1](#91---max-model-len-and-the-two-bound-calculation) |
| `--max-num-seqs` | 64 | Hard cap on how many requests can be running at once. Set just above our 50 target with room for bursts. | [9.3](#93---max-num-seqs-and-the-two-limits-lowest-wins-pattern) |
| `--max-num-batched-tokens` | 2048 | Limits how much work the GPU does in one step. Small chunks = smoother tail latency, because long prompts can't hog the GPU and stall everyone else. | [9.6](#96---enable-chunked-prefill--max-num-batched-tokens--the-prefilldecode-tradeoff) |
| `--gpu-memory-utilization` | 0.92 | Tells vLLM it can use 92% of the GPU's memory (default 0.9). Small bump only — if too high, the server crashes outright. | [9.7](#97---gpu-memory-utilization-and-asymmetric-failure-modes) |
| `--enable-prefix-caching` | on | When two requests share the same starting text, the second one skips redoing that work. Huge win because all 3 of our agent's calls share the schema. | [9.5](#95---enable-prefix-caching--the-agent-workload-superpower) |
| `--enable-chunked-prefill` | on | Splits long prompt-processing across multiple steps so other requests don't get stalled waiting. Pairs with `--max-num-batched-tokens`. | [9.6](#96---enable-chunked-prefill--max-num-batched-tokens--the-prefilldecode-tradeoff) |

### One sentence per design number

- **5 s SLO** → per-call LLM budget ≈ **1.5 s** (5 ÷ 3 calls).
- **10 RPS sustained** with 5 s latency → **~50 concurrent calls** at steady state (Little's Law).
- **31 B parameters, fp8** → weights take **~31 GB** of 80 GB.
- **0.92 utilization** → vLLM pool is **~73.6 GB**, leaving **~39 GB for KV cache** after weights and workspace.
- **8 K context cap + fp8 KV** → per request peak ~395 MB → **~100-request KV ceiling** with comfortable slack at the 50 target.

### Reading order if you forget what something does

1. Glance at the table above for the value and one-sentence why.
2. If the one-sentence isn't enough, follow the link to the full 9.x section for the bounds calculation, mechanical explanation, and one-liner rationale for REPORT.md.
3. If you need to defend the choice in conversation, the "What's at stake" paragraph at the top of each 9.x section is the soundbite.

---

## 11. Phase 3 — Agent design (Round 1)

Phase 3 is the actual product code — a LangGraph state machine that turns a question into SQL, runs it, verifies the result, and revises if needed. Round 1 was about the *design* — answering enough questions that the prompts (Round 2) and node implementations (Round 3) write themselves.

### 11.1 The graph shape and the node pattern

```
START → attach_schema → generate_sql → execute → verify → [router]
                                                            │
                                                  "end"     ┴     "revise"
                                                    │                │
                                                   END    revise → execute → verify → [router again]
```

The router is called after **every** verify. So if verify runs N times, the router is called N times.

#### The LLM-node pattern (worked out from `generate_sql_node`)

Every LLM-calling node follows the same shape. From `generate_sql_node` at `agent/graph.py:84–106`:

1. Build messages from `prompts.py` constants — system message + user message with placeholder substitution via `.format(...)`.
2. Get the client from `llm()` — a `ChatOpenAI` wrapper around the HTTP client pointing at vLLM (or any OpenAI-compatible endpoint via `VLLM_BASE_URL`).
3. Call `.invoke(messages)` — sends the chat-completion request, returns an `AIMessage` with `.content`.
4. **Parse defensively** — never trust the model's surface format. `generate_sql_node` uses `_extract_sql()` to strip markdown fences and prose. The verifier will need similar defensive JSON parsing.
5. Return a **partial state dict** — only the fields the node *changed* (e.g., `{"sql": ..., "iteration": ..., "history": ...}`). LangGraph merges this into `AgentState`. Unmentioned fields are left alone.
6. **Bump `iteration`** — both `generate_sql_node` and `revise_node` bump it. `iteration` counts *all* SQL-producing LLM calls, not just revisions, so `MAX_ITERATIONS = 3` means "3 attempts total."

#### Why "partial state dict" matters

The convention forces explicit thinking: each node has to declare what it changes. This makes graph edits safer (you can't accidentally clobber a field by being silent about it) and trace-friendlier (the diff between states tells you exactly what each node did).

### 11.2 Failure taxonomy for the verifier

The verifier exists to catch the cases where the SQL ran "successfully" in some narrow sense but the result doesn't actually answer the question. Four distinct failure modes, from easiest-to-catch to hardest:

| # | Failure mode | Field on `ExecutionResult` | Example `verify_issue` |
|---|---|---|---|
| 1 | SQL errored | `execution.ok == False` | "SQL execution failed: no such column 'patient_age'" |
| 2 | Zero rows when rows expected | `execution.row_count == 0` | "Query returned 0 rows, but the question asks for a list of customers — likely WHERE clause too aggressive." |
| 3 | Wrong columns | inspect `execution.columns` vs `state.question` | "Question asks for customer names, but returned 'customer_id' instead." |
| 4 | **Semantic mismatch** | no syntactic signal — needs LLM judgment | "Returns total student count, but question asks specifically for female students — missing gender filter." |

The fourth — **semantic mismatch** — is the hardest and the most important. Cases like:

- *"How many female students?"* → SQL returns total count, no gender filter.
- *"Top 5 most expensive products"* → SQL sorts by `units_sold` instead of `price`.
- *"Revenue in 2023"* → SQL sums all revenue with no year filter.

Failures 1–3 have syntactic signals you could catch with simple Python checks. Semantic mismatch is **the failure mode that justifies using an LLM as the verifier** — it requires reading the question and the result *together* and reasoning about whether they match.

#### Design implication

The verify prompt needs to put the model in a position to make this judgment. Concretely:

- Show it the question
- Show it the executed result (via `execution.render()`)
- Ask explicitly: *"does this result plausibly answer the question?"*
- Get back a structured `{ok: bool, issue: str}` it can return

`ExecutionResult.render()` is already shaped for prompt context — handles error vs zero-rows vs rows preview in one method. Use it; don't re-format rows in the prompt.

### 11.3 Loop dynamics and termination

Three scenarios trace the router's behavior:

| Scenario | Final `iteration` | Router calls (in order) | MAX_ITERATIONS reached? |
|---|---|---|---|
| **A**: First try passes | 1 | `"end"` | No |
| **B**: First fails, revise on iter 2 passes | 2 | `"revise"` → `"end"` | No |
| **C**: All 3 attempts fail | 3 | `"revise"` → `"revise"` → `"end"` (forced) | **Yes** |

The crucial line: in Scenario C, even though `verify_ok` is still `False` at iteration 3, the router *must* return `"end"`. The check on `iteration >= MAX_ITERATIONS` is what guarantees the graph terminates — without it, a stubborn verify could loop forever.

#### The router becomes trivial

The design question forces the implementation:

```python
def route_after_verify(state: AgentState) -> str:
    if state.verify_ok:                         return "end"
    if state.iteration >= MAX_ITERATIONS:       return "end"
    return "revise"
```

Three lines. That's the whole router.

#### What gets returned to the user in Scenario C

Read `agent/server.py:67–97`. The agent returns whatever the *last attempt's* execution produced — regardless of what verify thought.

| Last-attempt execution | Response |
|---|---|
| SQL errored | `ok=False, error=...` (agent admits defeat) |
| SQL ran cleanly, verify rejected the rows | `ok=True, rows=[...]` (agent serves rows it doesn't trust) |

**`AnswerResponse.ok` means "SQL executed without a database error" — NOT "verify approved this answer."** That's a design choice with a tradeoff:

- **Pro:** user always gets some answer.
- **Con:** user has no signal that verify rejected this one.

A defensible improvement (worth flagging in Phase 7 "what I'd do with more time"): surface `verify_ok` or `iteration` or `verify_issue` in `AnswerResponse` so the caller can detect low-confidence answers.

### 11.4 What revise needs to see (state field selection)

`AgentState` (`graph.py:42–54`) has nine fields. Not all of them belong in the revise prompt.

| Field | Include in revise prompt? | Why |
|---|---|---|
| `question` | ✅ | What the user wanted. |
| `schema` | ✅ | Tables/columns available. |
| `sql` | ✅ | The failing attempt. |
| `execution` | ✅ (rendered) | What happened when we ran it. Use `execution.render()`. |
| `verify_issue` | ✅ | The verifier's diagnosis — the load-bearing *new* signal. |
| `db_id` | ❌ | `schema` already encodes everything useful. |
| `iteration` | ❌ | Loop counter for the router; LLM doesn't need it. |
| `verify_ok` | ❌ | Always `False` here (we wouldn't be in revise otherwise). |
| `history` | ❌ (for v1) | See below. |

#### The `history` debate

`history` logs all prior nodes' outputs. Including it could help avoid loops (*"we tried X already, try something else"*). But:

- Most recent attempt is already fully captured in `sql`, `execution`, `verify_issue`.
- Older entries are duplicative or stale.
- Extra tokens hurt prefix-cache hit rate (Phase 1 habit #17) and add noise.
- Max 3 attempts means cross-iteration learning has limited payoff.

**Recommendation:** leave `history` out of v1. If Phase 6 evidence shows multi-iteration revisions repeating the same mistakes, revisit. Avoid prematurely complicating the prompt.

### 11.5 Round 1 design summary

#### Verify node

- **Inputs:** `question`, `execution.render()`
- **Output:** `{"verify_ok": bool, "verify_issue": str}` partial state dict
- **Job:** classify into one of the four failure modes (errored / zero rows / wrong columns / semantic mismatch), or pass.
- **Critical implementation detail:** ask the model for JSON; parse it defensively (it may come fenced, with prose around it, or with sloppy formatting).

#### Revise node

- **Inputs:** `question`, `schema`, `sql`, `execution.render()`, `verify_issue`
- **Excluded:** `db_id`, `iteration`, `verify_ok`, `history`
- **Output:** `{"sql": new_sql, "iteration": state.iteration + 1, "history": ...}` partial state dict
- **Pattern:** same shape as `generate_sql_node`, but the prompt also surfaces the prior attempt + verifier diagnosis.

#### Router

```python
if state.verify_ok:                  return "end"
if state.iteration >= MAX_ITERATIONS: return "end"
return "revise"
```

#### Prompt structure must be prefix-cache-friendly

Phase 1's `--enable-prefix-caching` only pays off if the shared tokens come at the **front** of every prompt:

```
[system prompt — same every call]
[schema — same per DB]
[question — same per agent request]
[per-iteration content goes LAST: failing SQL, exec result, verify_issue]
```

This makes the first ~1900 tokens cacheable across all three calls within one agent request — the 10–20× prefill speedup the Phase 1 report claims. **The Round 2 prompt-writing must respect this ordering.**

#### Habits this round added

- **The node return is a partial state dict, not a full one.** Be explicit about what each node changes; let LangGraph merge.
- **Never trust the model's surface format.** Parse defensively — extract SQL from fences in `generate_sql`/`revise`, parse JSON forgivingly in `verify`.
- **For loops with an iteration cap, the termination check must be unconditional.** Even if the "natural" exit condition (verify_ok) is never satisfied, the cap forces termination. Without it, the graph can deadlock.
- **Prompt content selection is also tokens-budget management.** Excluding a field isn't censorship — it's reducing noise, preserving cacheable prefix, and focusing the model's attention on signal.
