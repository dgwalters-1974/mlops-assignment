# Phase 1 — Serving configuration

## Workload framing

The agent makes ~3 sequential vLLM calls per user request. The platform SLO is P95 end-to-end < 5 s at 10+ RPS sustained, which decomposes into two design numbers: a **per-call latency budget of ~1.5 s** (5 s ÷ 3 calls, with a small buffer for sqlite + Python overhead) and a **concurrency target of ~50 in-flight vLLM calls**. The flags below exist to make both feasible on one 80 GB H100.

The model architecture (verified against [`config.json`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json)) sharpens these: 31 B params total / 3.3 B active per token (8-of-128 expert routing), 48 layers, GQA with 4 KV heads × head_dim 128, 256 K native context. MoE makes decode compute cheap, but all 31 B still live in VRAM (router decides per-token at inference). Per-token KV cost is exactly 96 KB at bf16 / 48 KB at fp8 — combined with the 50-concurrent target, this is what makes some flags below structurally required, not optional.

## Configuration and rationale

See `scripts/start_vllm.sh` for the launch command. Eight non-default flags:

**`--quantization fp8`** — bf16 weights take 62 GB of 80 GB, leaving ~6 GB for KV at default 0.9 utilization (`0.9 × 80 − 62 − ~4 GB workspace`); nowhere near enough for 50 concurrent, so bf16 makes the SLO infeasible. fp8 halves weights to 31 GB and runs at ~2× speed on H100's native fp8 tensor cores — a Hopper-specific win not available on Ampere.

**`--kv-cache-dtype fp8`** — fp8 weights alone leave ~0 GB of slack at peak 50 concurrent; any burst triggers preemption (vLLM evicting in-flight sequences, blowing P99). fp8 KV halves per-token KV (96 KB → 48 KB), restoring ~19 GB of slack and raising the concurrency ceiling from ~50 to ~100.

**`--max-model-len 8192`** — native 256 K is irrelevant when real prompts are ≤3 K. Capping at 8 K shrinks vLLM's worst-case-per-request provisioning by exactly 32× (262,144 / 8192), so the admission planner fits 50 concurrent: at fp8 KV, 50 × 8192 × 48 KB ≈ 20 GB (fits in the ~39 GB pool) vs 50 × 262 144 × 48 KB = 630 GB at native (wouldn't fit even 4 concurrent). 8192 is the smallest power-of-2 above worst case real prompt.

**`--max-num-seqs 64`** — scheduler-level cap on running-batch size, independent of KV. Below 50 makes the SLO impossible; far above the KV-derived ceiling (~100 with fp8 KV) wastes scheduler overhead. 64 = 50 target + ~28% burst headroom,rounded to power of 2.

**`--max-num-batched-tokens 2048`** — at 50 concurrent decoders with new ~3 K prefills arriving every ~33 ms, an unchunked prefill stalls every decode in its batch step — head-of-line blocking that spikes P99. 2048 splits 3 K prefills into 2 chunks that interleave with decode work. Trades slight prefill slowdown for P95 stability.

**`--gpu-memory-utilization 0.92`** — dedicated H100, no competing tenants, so a modest bump from default 0.9 buys ~1.6 GB more KV pool (~3–4 more concurrent requests' headroom). Increment is small because the failure mode is binary OOM, not graceful.

**`--enable-prefix-caching`** — verify/revise calls share a ~1900-token prefix (system + schema + question) with generate_sql; across requests on the same DB the ~1500-token schema is identical. Cache hits → ~10–20× prefill speedup on calls 2 and 3, zero quality cost. Deterministic prefix guaranteed by `@lru_cache`-stable schema rendering in `agent/schema.py`. Single biggest one-flag P95 lever for this workload.

**`--enable-chunked-prefill`** — companion to `--max-num-batched-tokens`; without it the per-step token budget can't be enforced across long prefills and head-of-line blocking returns. Default-on in current vLLM.

## Risks and detection

- **fp8 weight + fp8 KV quality regression** — detect via Phase 5 execution-accuracy on BIRD; mitigate by swapping to a pre-calibrated FP8 HF variant or falling back to bf16 KV (accepting tighter pool slack).
- **OOM at 0.92 utilization** — detect via vLLM process crash under Phase 6 load; mitigate by dropping to 0.9.
- **Prefill starvation at `max-num-batched-tokens 2048`** — detect via Grafana showing prefill queue growth while decode steps go underused; mitigate by bumping to 4096.

## What Phase 6 will likely revise

Starting config derived from SLO sketch, not observed load. Most likely to iterate: `--max-num-batched-tokens` (prefill/decode balance is the most workload-sensitive lever), `--gpu-memory-utilization` (push higher if no OOM and KV pressure remains binding), `--max-num-seqs` (raise if scheduler cap binds before KV).

## A note on BIRD's evidence field

Per task-author guidance in the course chat (Gleb Berjoskin, 8 Jun 2026), the BIRD `evidence` hints are intentionally not surfaced into the agent's prompts. The scaffolding's `scripts/load_data.py` drops the field; we preserve that behaviour. This caps achievable execution accuracy on databases with cryptic column names (e.g., `financial.A##` codes), since the model cannot learn from prompt context that `A15` means "crimes in 1995." Phase 5 baseline numbers should be read against this constraint; another participant reported ~33% without evidence vs ~53% with on the same model, so the gap is real and expected.
