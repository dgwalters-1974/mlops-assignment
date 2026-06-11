"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import httpx

# Matches MAX_ITERATIONS in agent/graph.py. Per-iteration pass rates run 1..this.
MAX_ITERATIONS = 3

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question.

    Calls the agent, then for each SQL the agent produced (generate_sql +
    every revise) runs it against the target DB and compares the row set
    against the gold SQL's row set using execution accuracy.

    Returns a dict with:
        question, db_id, gold_sql      — the eval inputs
        agent_ok, agent_error          — was the HTTP call successful?
        gold_ok, gold_error            — does the gold SQL even run?
        iterations_used                — how many SQL attempts the agent made
        final_sql, final_correct       — last attempt's SQL and whether it matched
                                         (final_correct is None if unscorable)
        per_iteration: list of         — one entry per agent attempt
            {iteration, sql, correct, exec_ok, error}
        wall_clock_seconds             — agent HTTP round-trip time
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    # Run gold once so we know what we're comparing against (and whether
    # the gold even works — broken gold = unscorable, not the agent's fault).
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    result: dict = {
        "question": question["question"],
        "db_id": db_id,
        "gold_sql": gold_sql,
        "agent_ok": False,
        "agent_error": None,
        "gold_ok": gold_ok,
        "gold_error": gold_err,
        "iterations_used": 0,
        "final_sql": "",
        "final_correct": None,
        "per_iteration": [],
        "wall_clock_seconds": 0.0,
    }

    payload = {
        "question": question["question"],
        "db": db_id,
        "tags": {"experiment": "eval-baseline", "db_id": db_id}, # Tags are passed as metadata rather than Langfuse tags - use for filtering in Phase 6
    }
    t0 = time.monotonic()
    try:
        resp = httpx.post(agent_url, json=payload, timeout=120.0)
    except Exception as e:  # noqa: BLE001
        result["agent_error"] = f"{type(e).__name__}: {e}"
        result["wall_clock_seconds"] = time.monotonic() - t0
        return result
    result["wall_clock_seconds"] = time.monotonic() - t0

    if resp.status_code != 200:
        result["agent_error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
        return result

    body = resp.json()
    result["agent_ok"] = True
    result["iterations_used"] = body.get("iterations", 0)
    result["final_sql"] = body.get("sql", "")

    # Pull the SQL string from each generate_sql / revise entry in history.
    # These appear in iteration order, so list index + 1 == iteration number.
    history = body.get("history", []) or []
    per_iter_sqls = [
        h.get("sql", "")
        for h in history
        if h.get("node") in ("generate_sql", "revise")
    ]

    for i, sql in enumerate(per_iter_sqls, start=1):
        exec_ok, pred_rows, exec_err = run_sql(db_id, sql)
        # Correctness requires both queries executed AND row sets match.
        # If gold is broken we still record per-iter exec status but score as False
        # (final_correct=None below signals "don't include in pass-rate denominator").
        correct = matches(gold_rows, pred_rows) if gold_ok and exec_ok else False
        result["per_iteration"].append({
            "iteration": i,
            "sql": sql,
            "correct": bool(correct),
            "exec_ok": exec_ok,
            "error": exec_err,
        })

    # Final correctness: last attempt's correctness if scorable, else None.
    if not gold_ok:
        result["final_correct"] = None
    elif result["per_iteration"]:
        result["final_correct"] = result["per_iteration"][-1]["correct"]
    else:
        # Agent responded but produced no SQL attempts — count as incorrect.
        result["final_correct"] = False

    return result


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n_total = len(results)
    # Scorable = both the gold SQL runs cleanly AND the agent responded.
    # These are the only questions that contribute to pass-rate denominators.
    scorable = [r for r in results if r["gold_ok"] and r["agent_ok"]]
    n_scorable = len(scorable)
    agent_errors = sum(1 for r in results if not r["agent_ok"])
    gold_errors = sum(1 for r in results if not r["gold_ok"])

    # Overall pass rate = final-attempt correctness across scorable questions.
    n_correct_final = sum(1 for r in scorable if r["final_correct"])
    overall = (n_correct_final / n_scorable) if n_scorable else 0.0

    # Per-iteration pass rate WITH carry-forward.
    # For each scorable question and each k in 1..MAX_ITERATIONS:
    #   - if k <= attempts: that attempt's correctness
    #   - else: carry forward the last attempt's correctness (agent already stopped)
    per_iter_correct = {k: 0 for k in range(1, MAX_ITERATIONS + 1)}
    for r in scorable:
        attempts = r["per_iteration"]
        if not attempts:
            continue
        last_correct = False
        for k in range(1, MAX_ITERATIONS + 1):
            if k <= len(attempts):
                last_correct = attempts[k - 1]["correct"]
            # else: carry forward — last_correct stays as it was at attempts[-1]
            if last_correct:
                per_iter_correct[k] += 1
    per_iter_pass_rate = {
        str(k): (per_iter_correct[k] / n_scorable) if n_scorable else 0.0
        for k in range(1, MAX_ITERATIONS + 1)
    }

    # Termination distribution: how many questions stopped at iter 1, 2, 3
    # (over questions where the agent actually answered).
    iter_dist = Counter(r["iterations_used"] for r in results if r["agent_ok"])

    # Per-DB pass rate (final correctness over scorable per DB).
    per_db_correct: Counter[str] = Counter()
    per_db_total: Counter[str] = Counter()
    for r in scorable:
        per_db_total[r["db_id"]] += 1
        if r["final_correct"]:
            per_db_correct[r["db_id"]] += 1
    per_db_pass_rate = {
        db: per_db_correct[db] / per_db_total[db]
        for db in per_db_total
    }

    # Latency stats over questions where the agent responded.
    wall_clocks = [r["wall_clock_seconds"] for r in results if r["agent_ok"]]
    mean_wall = (sum(wall_clocks) / len(wall_clocks)) if wall_clocks else 0.0

    return {
        "n_questions": n_total,
        "n_scorable": n_scorable,
        "agent_errors": agent_errors,
        "gold_errors": gold_errors,
        "overall_pass_rate": overall,
        "per_iteration_pass_rate": per_iter_pass_rate,
        "iteration_distribution": {str(k): v for k, v in sorted(iter_dist.items())},
        "per_db_pass_rate": per_db_pass_rate,
        "mean_wall_clock_seconds": mean_wall,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
