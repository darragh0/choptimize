#!/usr/bin/env python3

"""Semantic analysis of prompt-code pairs via OpenAI-compatible LLM server."""

from __future__ import annotations

import json
import os
import threading
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from random import uniform
from time import sleep
from typing import TYPE_CHECKING, Final, Literal, cast, get_args

import pandas as pd
from openai import BadRequestError, OpenAI
from rich_argparse import RichHelpFormatter
from utils.cache import CACHE_DIR
from utils.console import cerr, cout
from utils.display import show_df_overview
from utils.progress import tracked
from utils.types import CodeSemEval, PromptSemEval, Uint

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from utils.types import SemanticEvalRow, SyntaxEvalRow


type Dim = Literal["clarity", "specificity", "completeness", "correctness", "robustness", "readability", "efficiency"]

DIMS: Final[set[Dim]] = set(get_args(Dim.__value__))
DEFAULT_MODEL_VLLM: Final = "google/gemma-3-27b-it"
DEFAULT_HOST_VLLM: Final = "http://localhost:8000"
MAX_RETRIES: Final[Uint] = 3
DEFAULT_PARALLEL: Final[Uint] = 1

JSON_SCHEMA: Final[dict] = {
    "type": "object",
    "properties": {dim: {"type": "integer", "minimum": 1, "maximum": 5} for dim in get_args(Dim.__value__)},
    "required": sorted(get_args(Dim.__value__)),
    "additionalProperties": False,
}

_SYSPROMPT_PATH: Final = Path(__file__).parent / "semantic.py.prompt.xml"


def _load_system_prompt() -> str:
    """Try load grading rubric system prompt from file."""
    try:
        return _SYSPROMPT_PATH.read_text()
    except FileNotFoundError:
        cerr(f"system prompt not found: [cyan]{_SYSPROMPT_PATH}[/]", exit_code=1)
    except PermissionError:
        cerr(f"cannot read system prompt: [cyan]{_SYSPROMPT_PATH}[/] (permission denied)", exit_code=1)
    raise RuntimeError("unreachable")


SYSTEM_PROMPT: Final = _load_system_prompt()


def setup_client(host: str) -> OpenAI:
    """Connect to LLM server (Ollama or vLLM) and verify it is reachable."""
    try:
        client = OpenAI(base_url=f"{host}/v1", api_key="none")
        client.models.list()
    except Exception as e:
        cerr(f"Cannot connect to LLM server at [cyan]{host}[/] -- is it running?", exit_code=1)
        raise RuntimeError("unreachable") from e
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


def score_row(client: OpenAI, row: SyntaxEvalRow, model: str) -> dict:
    """Call LLM & extract JSON scores."""

    meow = (
        "<INSTRUCTIONS>Grade each dimension (1-5) following the rubric exactly. "
        "Start at the anchor score of 3 and adjust with evidence. Apply mandatory penalties.</INSTRUCTIONS>"
        f"<USER_PROMPT>{row['prompt']}</USER_PROMPT>"
        f"<LLM_RESPONSE>{row['response']}</LLM_RESPONSE>"
        f"<LLM_CODE>{row['code']}</LLM_CODE>"
        f"<SYNTAX_REPORT>{build_syntax_summary(row)}</SYNTAX_REPORT>"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": meow},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "eval", "strict": True, "schema": JSON_SCHEMA},
        },
        max_tokens=512,
    )

    txt = resp.choices[0].message.content
    if txt is None:
        msg = "LLM returned None"
        raise ValueError(msg)

    js: dict = json.loads(txt)
    for k in DIMS:
        js[k] = max(1, min(5, int(js[k])))
    return js


def process_row(client: OpenAI, row: SyntaxEvalRow, model: str) -> SemanticEvalRow | None:
    """Evaluate single row. Returns None if skipped."""

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            raw = score_row(client, row, model)
            break
        except BadRequestError:
            cerr(f"skip {row['id'][:8]}… — input too long")
            return None
        except Exception as e:  # noqa: BLE001
            last_err = e
            delay = min(30, 2 ** (attempt + 1)) * uniform(0.75, 1.25)  # noqa: S311
            cerr(f"row {row['id'][:8]}… retry {attempt + 1}/{MAX_RETRIES}: {type(e).__name__}: {e}")
            sleep(delay)
    else:
        cerr(f"skip {row['id'][:8]}… — failed after {MAX_RETRIES} retries (last: {last_err})")
        return None

    return cast("SemanticEvalRow", {**row, **{dim: raw[dim] for dim in DIMS}})


def load_checkpoint(path: Path) -> list[SemanticEvalRow]:
    """Load completed rows from JSONL checkpoint file."""
    if not path.exists():
        return []

    records: list[SemanticEvalRow] = []
    with path.open() as f:
        for line in f:
            if not (line := line.strip()):
                continue
            try:
                records.append(cast("SemanticEvalRow", json.loads(line)))
            except json.JSONDecodeError:
                break
    return records


@contextmanager
def checkpoint_writer(path: Path) -> Generator[Callable[[SemanticEvalRow], None]]:
    """Yield function to append scored rows to checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:

        def write(row: SemanticEvalRow) -> None:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())

        yield write


def show_oview(df: pd.DataFrame) -> None:
    cout("Semantic Analysis Summary:")

    dims = [*PromptSemEval.__annotations__, *CodeSemEval.__annotations__]
    for col in dims:
        mean = df[col].mean()
        med = df[col].median()
        cout(f"  [dim]{col:<20}[/] mean={mean:.2f}  median={med:.2f}")


def _shard_paths(shard_id: int, shard_total: int) -> tuple[Path, Path]:
    """Return (cache_path, checkpoint_path) for a given shard."""
    tag = f"shard_{shard_id}_of_{shard_total}"
    return (
        CACHE_DIR / f"semantic_eval.{tag}.parquet",
        CACHE_DIR / f"semantic_eval.{tag}.checkpoint.jsonl",
    )


def _score_rows(
    client: OpenAI,
    remaining: list[SyntaxEvalRow],
    done: list[SemanticEvalRow],
    checkpoint_path: Path,
    model: str,
    parallel: int,
) -> int:
    """Score remaining rows, appending to done in-place. Returns skipped count."""
    skipped = 0
    write_lock = threading.Lock()

    total = len(remaining) + len(done)
    if parallel <= 1:
        with checkpoint_writer(checkpoint_path) as write:
            for _, row in tracked(remaining, "Scoring semantics", total=total, completed=len(done)):
                result = process_row(client, row, model)
                if result is None:
                    skipped += 1
                    continue
                done.append(result)
                write(result)
    else:
        cout(f"Running {parallel} parallel workers")
        with checkpoint_writer(checkpoint_path) as write, ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(process_row, client, row, model): row for row in remaining}
            for _, future in tracked(
                as_completed(futures),
                "Scoring semantics",
                total=total,
                completed=len(done),
            ):
                result = future.result()
                if result is None:
                    skipped += 1
                    continue
                with write_lock:
                    done.append(result)
                    write(result)

    return skipped


def analyse_semantics(
    df: pd.DataFrame,
    model: str,
    host: str = DEFAULT_HOST_VLLM,
    shard: tuple[int, int] | None = None,
    parallel: int = 1,
) -> pd.DataFrame:
    """Run LLM-as-a-judge semantic analysis on each row."""

    if shard:
        shard_id, shard_total = shard
        cache_path, checkpoint_path = _shard_paths(shard_id, shard_total)
    else:
        cache_path = CACHE_DIR / "semantic_eval.parquet"
        checkpoint_path = CACHE_DIR / "semantic_eval.checkpoint.jsonl"

    def compute() -> pd.DataFrame:
        client = setup_client(host)
        all_rows = cast("list[SyntaxEvalRow]", df.to_dict("records"))

        if shard:
            shard_id, shard_total = shard
            all_rows = all_rows[shard_id - 1 :: shard_total]
            cout(f"Shard {shard_id}/{shard_total}: {len(all_rows):,} rows")

        done = load_checkpoint(checkpoint_path)
        done_ids = {r["id"] for r in done}
        remaining = [r for r in all_rows if r["id"] not in done_ids]

        skipped = _score_rows(client, remaining, done, checkpoint_path, model, parallel)

        if skipped:
            cerr(f"{skipped} rows skipped — excluded from output")

        done.sort(key=lambda r: r["id"])
        checkpoint_path.unlink(missing_ok=True)

        return pd.DataFrame(done)

    if cache_path.exists():
        result = pd.read_parquet(cache_path)
    else:
        result = compute()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path)
        cout(f"Cached to [cyan]{cache_path}[/]")
    cout()

    show_oview(result)
    cout()
    show_df_overview(result)
    cout()

    return result


def merge_shards() -> None:
    """Merge shard parquet files into final semantic_eval.parquet."""
    shard_files = sorted(CACHE_DIR.glob("semantic_eval.shard_*.parquet"))
    if not shard_files:
        cerr("no shard files found in [cyan]data/[/]", exit_code=1)

    cout(f"Merging {len(shard_files)} shard files:")
    dfs: list[pd.DataFrame] = []
    for f in shard_files:
        df = pd.read_parquet(f)
        cout(f"  {f.name}: {len(df):,} rows")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True).sort_values("id").reset_index(drop=True)
    dup_ids = merged[merged["id"].duplicated(keep=False)]
    if not dup_ids.empty:
        cerr(f"WARNING: {len(dup_ids)} duplicate IDs found — check shard overlap")

    out_path = CACHE_DIR / "semantic_eval.parquet"
    merged.to_parquet(out_path)
    cout(f"Wrote {len(merged):,} rows to [cyan]{out_path}[/]")

    for f in shard_files:
        f.unlink()
    # Clean up shard checkpoints too
    for f in CACHE_DIR.glob("semantic_eval.shard_*.checkpoint.jsonl"):
        f.unlink()
    cout("Cleaned up shard files")


def parse_shard(value: str) -> tuple[int, int]:
    """Parse 'K/N' shard argument."""
    parts = value.split("/")
    if len(parts) != 2:  # noqa: PLR2004
        msg = f"--shard must be K/N (e.g. 1/3), got: {value}"
        raise ValueError(msg)
    k, n = int(parts[0]), int(parts[1])
    if not (1 <= k <= n):
        msg = f"--shard K/N requires 1 <= K <= N, got: {k}/{n}"
        raise ValueError(msg)
    return k, n


def main() -> None:
    parser = ArgumentParser(description="Semantic analysis of prompt-code pairs", formatter_class=RichHelpFormatter)
    parser.add_argument("--sample", type=int, default=None, help="Random sample size (default: all rows)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_VLLM, help=f"Model name (default: {DEFAULT_MODEL_VLLM})"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST_VLLM,
        help=f"LLM server base URL -- Ollama or vLLM (default: {DEFAULT_HOST_VLLM})",
    )
    parser.add_argument("--shard", type=str, default=None, help="Shard specification K/N (e.g. [cyan]1/3[/])")
    parser.add_argument("--merge", action="store_true", help="Merges shard parquet files into final output and exit")
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        help=f"Concurrent workers (default: {DEFAULT_PARALLEL})",
    )

    args = parser.parse_args()
    if args.merge:
        merge_shards()
        return

    shard = parse_shard(args.shard) if args.shard else None

    syntax_fname = "syntax_eval.parquet"
    cache_path = CACHE_DIR / syntax_fname

    is_dev = os.getenv("CHOP__DEV") is not None
    has_cache = cache_path.exists() or is_dev

    if not has_cache:
        cerr(f"run [cyan]scripts/syntax.py[/] first -- missing [cyan]{syntax_fname}[/]", exit_code=1)

    df = (
        pd.read_parquet("hf://datasets/darragh0/prompt2code-static-analysis/syntax_eval.parquet")
        if is_dev
        else pd.read_parquet(cache_path)
    )

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        cout(f"Sampled {args.sample:,} rows (seed={args.seed})")

    analyse_semantics(df, model=args.model, host=args.host, shard=shard, parallel=args.parallel)


if __name__ == "__main__":
    from utils.cache import graceful_exit

    with graceful_exit("semantic analysis stopped"):
        main()
