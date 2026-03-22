#!/usr/bin/env python3

"""Semantic analysis of prompt-code pairs via Ollama LLM."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast, get_args

from ollama import Client
from pandas import DataFrame, read_parquet
from utils.cache import CACHE_DIR, parquet_cache
from utils.console import cerr, cout
from utils.display import show_df_overview
from utils.progress import tracked
from utils.types import CodeSemEval, PromptSemEval, Uint

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from utils.types import SemanticEvalRow, SyntaxEvalRow


type Dim = Literal["clarity", "specificity", "completeness", "correctness", "robustness", "readability", "efficiency"]

DIMS: Final[set[Dim]] = set(get_args(Dim.__value__))
MODEL: Final = "qwen3-coder:30b"
MAX_SINGLE_RETRY: Final[Uint] = 10
_SYSPROMPT_PATH: Final = Path(__file__).parent / "semantics-prompt.xml"


def _load_system_prompt() -> str:
    """Load grading rubric system prompt from file."""
    try:
        return _SYSPROMPT_PATH.read_text()
    except FileNotFoundError:
        cerr(f"system prompt not found: [cyan]{_SYSPROMPT_PATH}[/]", exit_code=1)
    except PermissionError:
        cerr(f"cannot read system prompt: [cyan]{_SYSPROMPT_PATH}[/] (permission denied)", exit_code=1)
    raise RuntimeError("unreachable")


SYSTEM_PROMPT: Final = _load_system_prompt()


def json_find_and_loads(txt: str) -> dict:
    """Find, extract & load JSON from str."""

    snip = f"{txt[: -(len(txt) // 2)]!r}"
    end = txt.rfind("}")
    if end == -1:
        msg = f"[JSON] No close brace (`}}`): ... {snip}"
        raise ValueError(msg)

    start = txt.rfind("{", 0, end)
    if start == -1:
        msg = f"[JSON] No open brace (`{{`): ... {snip}"
        raise ValueError(msg)

    try:
        js = json.loads(txt[start : end + 1])
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON: {snip}"
        raise ValueError(msg) from e

    return js


def check_json_fmt(js: dict) -> None:
    """Check JSON vs. required format (also clamp int overflows in-place)."""
    missing_keys = set(DIMS) - js.keys()
    if missing_keys:
        msg = f"Missing keys: {sorted(missing_keys)!r}"
        raise ValueError(msg)

    for k, v in js.items():
        if not isinstance(v, (int, float)):
            msg = f"Non-numeric value for key {k!r}: {v!r}"
            raise TypeError(msg)
        js[k] = max(1, min(5, int(v)))


def get_llm_json(txt: str | None) -> dict:
    """Extract last JSON object from CoT (Chain of Thought) response."""
    if txt is None:
        msg = "LLM returned None"
        raise ValueError(msg)

    js = json_find_and_loads(txt)
    check_json_fmt(js)
    return js


def check_ollama(model: str) -> Client:
    """Verify Ollama running & model available."""
    try:
        client = Client()
        available = client.list().models
    except Exception as e:
        cerr("Ollama not running -- start it with [cyan]ollama serve[/]", exit_code=1)
        raise RuntimeError("unreachable") from e

    has_model = any(False if m.model is None else m.model.startswith(model) for m in available)
    if not has_model:
        cerr(f"model [cyan]{model}[/] not found -- pull it with [cyan]ollama pull {model}[/]", exit_code=1)

    return client


def build_syntax_summary(row: SyntaxEvalRow) -> str:
    """Build human-readable syntax report from static analysis fields."""
    return (
        f"parseable={row['parseable']} | lines={row['lines']}"
        f" | ruff_errors={row['ruff_errors']} | ruff_warnings={row['ruff_warnings']}"
        f" | ruff_flake8={row['ruff_flake8']} | ruff_bugbear={row['ruff_bugbear']}"
        f" | ruff_security={row['ruff_security']}"
        f" | complexity={row['complexity']:.1f} | maintainability={row['maintainability']:.1f}"
    )


def score_row(client: Client, row: SyntaxEvalRow) -> dict:
    """Call LLM & extract JSON scores."""

    meow = (
        "<INSTRUCTIONS>Grade each dimension (1-7) following the rubric exactly. "
        "Start at the anchor score of 3 and adjust with evidence. Apply mandatory penalties. "
        "Output the final scores as the LAST thing in your response as a JSON object (no backticks). "
        'Format: {"clarity":N,"specificity":N,"completeness":N,"correctness":N,'
        '"robustness":N,"readability":N,"efficiency":N}</INSTRUCTIONS>'
        f"<USER_PROMPT>{row['prompt']}</USER_PROMPT>"
        f"<LLM_RESPONSE>{row['response']}</LLM_RESPONSE>"
        f"<LLM_CODE>{row['code']}</LLM_CODE>"
        f"<SYNTAX_REPORT>{build_syntax_summary(row)}</SYNTAX_REPORT>"
    )

    resp = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": meow},
        ],
    )

    return get_llm_json(resp.message.content)


def process_row(client: Client, row: SyntaxEvalRow) -> SemanticEvalRow | None:
    """Evaluate single row on all prompt + code dimensions."""

    for _ in range(MAX_SINGLE_RETRY):
        try:
            raw = score_row(client, row)
            break
        except (TypeError, ValueError):
            pass
    else:
        cerr(f"skipping row {row['id']} — failed after {MAX_SINGLE_RETRY} retries")
        return None

    return cast("SemanticEvalRow", {**row, **{dim: raw[dim] for dim in DIMS}})


def load_checkpoint(path: Path) -> list[SemanticEvalRow]:
    """Load completed rows from JSONL checkpoint file."""
    if not path.exists():
        return []

    records: list[SemanticEvalRow] = []
    with path.open() as f:
        records.extend([cast("SemanticEvalRow", json.loads(line.strip())) for line in f if line.strip()])
    return records


@contextmanager
def checkpoint_writer(path: Path) -> Generator[Callable[[SemanticEvalRow], None]]:
    """Yield function to append scored rows to checkpoint file."""
    with path.open("a") as f:

        def write(row: SemanticEvalRow) -> None:
            f.write(json.dumps(row) + "\n")
            f.flush()

        yield write


def show_oview(df: DataFrame) -> None:
    cout("Semantic Analysis Summary:")

    dims = [*PromptSemEval.__annotations__, *CodeSemEval.__annotations__]
    for col in dims:
        mean = df[col].mean()
        med = df[col].median()
        cout(f"  [dim]{col:<20}[/] mean={mean:.2f}  median={med:.2f}")


def analyse_semantics(df: DataFrame) -> DataFrame:
    """Run LLM-as-a-judge semantic analysis on each row."""

    cache_path = CACHE_DIR / "semantic_eval.parquet"
    checkpoint_path = CACHE_DIR / "semantic_eval.checkpoint.jsonl"

    def compute() -> DataFrame:
        client = check_ollama(MODEL)
        all_rows = cast("list[SyntaxEvalRow]", df.to_dict("records"))

        done = load_checkpoint(checkpoint_path)
        done_ids = {r["id"] for r in done}
        remaining = [r for r in all_rows if r["id"] not in done_ids]

        with checkpoint_writer(checkpoint_path) as write:
            for _, row in tracked(remaining, "Scoring semantics", total=len(all_rows), completed=len(done)):
                result = process_row(client, row)
                if result is None:
                    continue
                done.append(result)
                write(result)

        done.sort(key=lambda r: r["id"])
        checkpoint_path.unlink(missing_ok=True)
        return DataFrame(done)

    result = parquet_cache(cache_path, compute)
    cout()

    show_oview(result)
    cout()
    show_df_overview(result)
    cout()

    return result


def main() -> None:
    parser = ArgumentParser(description="Semantic analysis of prompt-code pairs")
    parser.add_argument("--sample", type=int, default=None, help="Random sample size (default: all rows)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    syntax_fname = "syntax_eval.parquet"
    cache_path = CACHE_DIR / syntax_fname
    if not cache_path.exists():
        cerr(f"run [cyan]scripts/syntax.py[/] first -- missing [cyan]{syntax_fname}[/]", exit_code=1)

    df = read_parquet(cache_path)

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        cout(f"Sampled {args.sample:,} rows (seed={args.seed})")

    analyse_semantics(df)


if __name__ == "__main__":
    from utils.cache import graceful_exit

    with graceful_exit("semantic analysis stopped"):
        main()
