"""Run evaluation prompts through Choptimize and save results."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.engine import Engine

if TYPE_CHECKING:
    from app.engine.types import LLMService

EVAL_DIR = Path(__file__).parent
PROMPTS_FILE = EVAL_DIR / "prompts.json"
RESULTS_FILE = EVAL_DIR / "results.json"

_MAX_RETRIES = 5


def _run_one(engine: Engine, p: dict[str, Any]) -> dict[str, Any]:
    pid, expected, prompt = p["id"], p["expected_quality"], p["prompt"]

    for attempt in range(_MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            r = engine.analyze(prompt, improve=True, show_raw=False)
            elapsed = time.perf_counter() - t0
        except ValueError as e:
            print(f"  ERROR: {e}")
            return {"id": pid, "expected_quality": expected, "prompt": prompt, "error": str(e)}
        except Exception as e:  # noqa: BLE001
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 65 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  ERROR: {err[:200]}")
            return {"id": pid, "expected_quality": expected, "prompt": prompt, "error": err[:200]}
        else:
            scores = {
                "clarity": r.scores.clarity.score,
                "specificity": r.scores.specificity.score,
                "completeness": r.scores.completeness.score,
                "overall": round(
                    (r.scores.clarity.score + r.scores.specificity.score + r.scores.completeness.score) / 3, 2
                ),
            }
            print(
                f"  scores: c={scores['clarity']} s={scores['specificity']} cp={scores['completeness']} "
                f"avg={scores['overall']} ({elapsed:.1f}s)"
            )
            return {
                "id": pid,
                "expected_quality": expected,
                "prompt": prompt,
                "source": p.get("source", "synthetic"),
                "dataset_scores": p.get("dataset_scores"),
                "scores": scores,
                "explanations": {
                    "clarity": r.scores.clarity.explanation,
                    "specificity": r.scores.specificity.explanation,
                    "completeness": r.scores.completeness.explanation,
                },
                "code_quality_outlook": r.scores.code_quality_outlook,
                "summary": r.scores.summary,
                "antipatterns": [ap.name for ap in r.detected_antipatterns],
                "techniques": [t.name for t in r.relevant_techniques],
                "improved_prompt": r.improvement.improved_prompt if r.improvement else None,
                "improvement_changes": [
                    {"dimension": c.dimension, "technique": c.technique_applied}
                    for c in (r.improvement.changes if r.improvement else [])
                ],
                "elapsed_s": round(elapsed, 2),
            }

    print("  ERROR: rate limit exhausted")
    return {"id": pid, "expected_quality": expected, "prompt": prompt, "error": "rate limit exhausted after retries"}


def run(service: LLMService, api_key: str | None = None, model: str | None = None) -> None:
    prompts = json.loads(PROMPTS_FILE.read_text())
    engine = Engine(service=service, model=model, api_key=api_key)
    results: list[dict[str, Any]] = []

    for p in prompts:
        print(f"[{p['id']}/{len(prompts)}] ({p['expected_quality']}) {p['prompt'][:60]}...")
        results.append(_run_one(engine, p))
        RESULTS_FILE.write_text(json.dumps(results, indent=2))

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    svc: LLMService = "ollama"
    key = None
    mdl = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("-s", "--service") and i + 1 < len(args):
            svc = args[i + 1]  # type: ignore[assignment]
            i += 2
        elif args[i] in ("-k", "--api-key") and i + 1 < len(args):
            key = args[i + 1]
            i += 2
        elif args[i] in ("-m", "--model") and i + 1 < len(args):
            mdl = args[i + 1]
            i += 2
        else:
            i += 1

    run(svc, key, mdl)
