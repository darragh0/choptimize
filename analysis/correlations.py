"""Statistical analysis: prompt quality vs. code quality correlations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pandas import read_parquet
from scipy import stats

from common.utils.cache import CACHE_DIR
from common.utils.console import cerr
from common.utils.display import pretty_path

if TYPE_CHECKING:
    from pandas import DataFrame

PROMPT_DIMS: Final = ("clarity", "specificity", "completeness")
CODE_DIMS: Final = ("correctness", "robustness", "readability", "efficiency")
SYNTAX_DIMS: Final = ("ruff_errors", "ruff_warnings", "ruff_flake8", "ruff_bugbear", "complexity", "maintainability")
OUTPUT_DIR: Final = CACHE_DIR / "analysis"
MIN_GROUP_SIZE: Final = 30
P_THRESHOLD: Final = 0.05


def load_data() -> DataFrame:
    """Load semantic evaluation parquet (contains all columns including syntax)."""
    semantic_path = CACHE_DIR / "semantic_eval.parquet"

    if not semantic_path.exists():
        cerr(f"missing {pretty_path(semantic_path)} — run the prerequisite scripts first", exit_code=1)

    return read_parquet(semantic_path)


def compute_correlations(df: DataFrame) -> dict:
    """Pearson correlations between prompt and code/syntax dimensions."""
    results: dict = {"prompt_vs_code": {}, "prompt_vs_syntax": {}}

    for pdim in PROMPT_DIMS:
        results["prompt_vs_code"][pdim] = {}
        for cdim in CODE_DIMS:
            r, p = stats.pearsonr(df[pdim], df[cdim])
            results["prompt_vs_code"][pdim][cdim] = {"r": round(r, 4), "p": round(p, 6)}

        results["prompt_vs_syntax"][pdim] = {}
        for sdim in SYNTAX_DIMS:
            if sdim in df.columns:
                r, p = stats.pearsonr(df[pdim], df[sdim])
                results["prompt_vs_syntax"][pdim][sdim] = {"r": round(r, 4), "p": round(p, 6)}

    return results


def compute_regression(df: DataFrame) -> dict:
    """Linear regression: composite prompt quality -> composite code quality."""
    prompt_score = df[list(PROMPT_DIMS)].mean(axis=1)
    code_score = df[list(CODE_DIMS)].mean(axis=1)

    slope, intercept, r, p, stderr = stats.linregress(prompt_score, code_score)
    return {
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(r**2, 4),
        "p_value": round(p, 6),
        "stderr": round(stderr, 4),
    }


def stratify_by_model(df: DataFrame) -> dict:
    """Per-model correlation between composite prompt and code quality."""
    results: dict = {}
    for model_name, group in df.groupby("model"):
        if len(group) < MIN_GROUP_SIZE:
            continue
        prompt_score = group[list(PROMPT_DIMS)].mean(axis=1)
        code_score = group[list(CODE_DIMS)].mean(axis=1)
        r, p = stats.pearsonr(prompt_score, code_score)
        results[str(model_name)] = {
            "n": len(group),
            "r": round(r, 4),
            "p": round(p, 6),
        }
    return results


def compute_descriptive_stats(df: DataFrame) -> dict:
    """Descriptive statistics for all scored dimensions."""
    all_dims = list(PROMPT_DIMS) + list(CODE_DIMS)
    result: dict = {}
    for dim in all_dims:
        col = df[dim]
        result[dim] = {
            "mean": round(col.mean(), 3),
            "median": round(col.median(), 3),
            "std": round(col.std(), 3),
            "min": int(col.min()),
            "max": int(col.max()),
            "distribution": {str(i): int((col == i).sum()) for i in range(1, 6)},
        }
    return result
