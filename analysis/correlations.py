"""Statistical analysis: prompt quality vs. code quality correlations."""

from __future__ import annotations

from typing import Final, TypedDict

import numpy as np
import pandas as pd
from common.utils.cache import CACHE_DIR
from scipy import stats

PROMPT_DIMS: Final = ["clarity", "specificity", "completeness"]
CODE_DIMS: Final = ["correctness", "robustness", "readability", "efficiency"]
SYNTAX_COLS: Final = (
    "ruff_errors",
    "ruff_warnings",
    "ruff_flake8",
    "ruff_bugbear",
    "ruff_security",
    "complexity",
    "maintainability",
)
OUTPUT_DIR: Final = CACHE_DIR / "analysis"
MIN_GROUP_SIZE: Final = 30


class ModelCorr(TypedDict):
    model: str
    n: int
    rho: float
    p_value: float
    mean_prompt_q: float
    mean_code_q: float


def descriptive_stats(df: pd.DataFrame) -> dict:
    """Distributions for all scored dims, + parseable rate & per-model counts."""
    result = {
        "dims": {},
        "parseable_rate": df["parseable"].mean(),
        "model_counts": df.groupby("model").size().to_dict(),
    }

    for dim in (*PROMPT_DIMS, *CODE_DIMS):
        col = df[dim]
        result["dims"][dim] = {
            "mean": round(col.mean(), 3),
            "median": round(col.median(), 3),
            "std": round(col.std(), 3),
            "min": int(col.min()),
            "max": int(col.max()),
            "distribution": {str(i): int((col == i).sum()) for i in range(1, 6)},
        }

    return result


def spearman_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """3x4 Spearman rho matrix (prompt dims x code dims) with Bonferroni-adjusted p-values.

    Returns (rho_df, p_adj_df). Both have PROMPT_DIMS as index, CODE_DIMS as columns.
    """
    n_tests = len(PROMPT_DIMS) * len(CODE_DIMS)  # 12
    rho = np.zeros((len(PROMPT_DIMS), len(CODE_DIMS)))
    p = np.zeros_like(rho)

    for i, pdim in enumerate(PROMPT_DIMS):
        for j, cdim in enumerate(CODE_DIMS):
            result = stats.spearmanr(df[pdim], df[cdim])
            rho[i, j] = result.statistic
            p[i, j] = result.pvalue

    idx, cols = PROMPT_DIMS, CODE_DIMS
    rho_df = pd.DataFrame(rho, index=idx, columns=cols)
    p_adj_df = pd.DataFrame((p * n_tests).clip(max=1.0), index=idx, columns=cols)
    return rho_df, p_adj_df


def prompt_dim_ranking(rho_df: pd.DataFrame) -> pd.DataFrame:
    """Rank prompt dims by mean Spearman rho across all code dims.

    Answers: which prompt characteristic correlates most strongly with code quality?
    Returns a DataFrame with columns: prompt_dim, mean_rho, sorted descending.
    """
    return (
        rho_df.mean(axis=1)
        .rename("mean_rho")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "prompt_dim"})
    )


def syntax_vs_semantic(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """4x7 Spearman rho matrix (code dims x syntax cols) with Bonferroni-adjusted p-values.

    Validates the LLM-as-judge: checks whether LLM-judged code quality scores
    agree with objective static-analysis metrics.
    """
    present = [c for c in SYNTAX_COLS if c in df.columns]
    n_tests = len(CODE_DIMS) * len(present)
    rho = np.zeros((len(CODE_DIMS), len(present)))
    p = np.zeros_like(rho)

    for i, cdim in enumerate(CODE_DIMS):
        for j, scol in enumerate(present):
            result = stats.spearmanr(df[cdim], df[scol])
            rho[i, j] = result.statistic
            p[i, j] = result.pvalue

    idx, cols = CODE_DIMS, present
    rho_df = pd.DataFrame(rho, index=idx, columns=cols)
    p_adj_df = pd.DataFrame((p * n_tests).clip(max=1.0), index=idx, columns=cols)
    return rho_df, p_adj_df


def composite_regression(df: pd.DataFrame) -> dict:
    """OLS + Spearman on composite prompt quality vs composite code quality."""
    prompt_q = df[PROMPT_DIMS].mean(axis=1)
    code_q = df[CODE_DIMS].mean(axis=1)

    ols = stats.linregress(prompt_q, code_q)
    sp = stats.spearmanr(prompt_q, code_q)

    return {
        "slope": round(ols.slope, 4),
        "intercept": round(ols.intercept, 4),
        "r_squared": round(ols.rvalue**2, 4),
        "p_value_ols": round(ols.pvalue, 6),
        "stderr": round(ols.stderr, 4),
        "spearman_rho": round(sp.statistic, 4),
        "spearman_p": round(sp.pvalue, 6),
    }


def per_model_correlation(df: pd.DataFrame) -> list[ModelCorr]:
    """Spearman rho per model (composite prompt → composite code quality).

    Returns a list of ModelCorr dicts sorted by rho descending.
    """
    rows: list[ModelCorr] = []
    for model_name, group in df.groupby("model"):
        if len(group) < MIN_GROUP_SIZE:
            continue
        prompt_q = group[PROMPT_DIMS].mean(axis=1)
        code_q = group[CODE_DIMS].mean(axis=1)
        result = stats.spearmanr(prompt_q, code_q)
        rows.append(
            ModelCorr(
                model=str(model_name),
                n=len(group),
                rho=round(result.statistic, 4),
                p_value=round(result.pvalue, 6),
                mean_prompt_q=round(prompt_q.mean(), 3),
                mean_code_q=round(code_q.mean(), 3),
            )
        )

    return sorted(rows, key=lambda r: r["rho"], reverse=True)


def kruskal_wallis(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis H-test: do code quality scores differ significantly across models?

    Justifies per-model stratification. One test per code dim + composite.
    """
    df = df.copy()
    df["code_composite"] = df[CODE_DIMS].mean(axis=1)
    results: dict = {}

    for dim in (*CODE_DIMS, "code_composite"):
        groups = [g[dim].to_numpy() for _, g in df.groupby("model") if len(g) >= MIN_GROUP_SIZE]
        h, p = stats.kruskal(*groups)
        results[dim] = {"H": round(float(h), 4), "p_value": round(float(p), 6)}

    return results
