#!/usr/bin/env python3

"""Run full correlation analysis on prompt2code-eval and print a summary."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, cast

from common.utils.cache import CACHE_DIR, graceful_exit, parquet_cache
from common.utils.console import cout
from common.utils.dataset import load_ds
from rich.table import Table

from .correlations import (
    CODE_DIMS,
    OUTPUT_DIR,
    PROMPT_DIMS,
    ModelCorr,
    composite_regression,
    descriptive_stats,
    kruskal_wallis,
    per_model_correlation,
    prompt_dim_ranking,
    spearman_matrix,
    syntax_vs_semantic,
)

if TYPE_CHECKING:
    import pandas as pd

DS_NAME: Final = "darragh0/prompt2code-eval"
DS_REVISION: Final = "1c01b1b582c8d929f24fc05a13e108ee31de8a0d"  # 2026-03-27


def _load_df() -> pd.DataFrame:
    return cast("pd.DataFrame", load_ds(DS_NAME, revision=DS_REVISION, split="test").to_pandas())


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _rho_color(rho: float) -> str:
    if rho >= 0.3:
        return "green"
    if rho <= -0.3:
        return "red"
    return "cyan"


def _p_color(p: float) -> str:
    return "green" if p < 0.05 else "dim"


def _fmt_p(p: float) -> str:
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


######################### Table Builders #########################


def _model_count_table(desc: dict) -> Table:
    tbl = Table("[dim]model[/]", "[dim]n[/]", box=None, padding=(0, 2))
    for model, n in sorted(desc["model_counts"].items(), key=lambda x: -x[1]):
        tbl.add_row(f"[green]{model}[/]", f"[yellow]{n}[/]")
    return tbl


def _matrix_table(rho_df: pd.DataFrame, p_adj_df: pd.DataFrame) -> Table:
    rho_arr = rho_df.to_numpy()
    p_arr = p_adj_df.to_numpy()
    tbl = Table("", *CODE_DIMS, box=None, padding=(0, 2))
    for i, pdim in enumerate(PROMPT_DIMS):
        cells = [
            f"[{_rho_color(rho_arr[i, code_dim])}]{rho_arr[i, code_dim]:.3f}[/][dim]{_sig_stars(p_arr[i, code_dim])}[/]"
            for code_dim in range(len(CODE_DIMS))
        ]
        tbl.add_row(pdim, *cells)
    return tbl


def _ranking_table(ranking: pd.DataFrame) -> Table:
    heads = (f"[dim]{thing}[/]" for thing in ("rank", "dimension", "mean rho"))
    tbl = Table(*heads, box=None, padding=(0, 2))
    for rank, row in enumerate(ranking.itertuples(index=False), start=1):
        rho = float(row.mean_rho)  # type: ignore[arg-type]
        tbl.add_row(f"[yellow]{rank}[/]", str(row.prompt_dim), f"[{_rho_color(rho)}]{rho:.3f}[/]")
    return tbl


def _regression_line(reg: dict) -> str:
    c = "cyan"
    return (
        f"slope=[{c}]{reg['slope']}[/]  "
        f"[yellow]R^2[/]=[{c}]{reg['r_squared']}[/]  "
        f"rho=[{_rho_color(reg['spearman_rho'])}]{reg['spearman_rho']}[/]  "
        f"p=[{_p_color(reg['spearman_p'])}]{_fmt_p(reg['spearman_p'])}[/]"
    )


def _model_corr_table(model_corr: list[ModelCorr]) -> Table:
    heads = (f"[dim]{thing}[/]" for thing in ("model", "n", "rho", "p"))
    tbl = Table(*heads, box=None, padding=(0, 2))
    for row in model_corr:
        tbl.add_row(
            f"[green]{row['model']}[/]",
            f"[yellow]{row['n']}[/]",
            f"[{_rho_color(row['rho'])}]{row['rho']:.3f}[/]",
            f"[{_p_color(row['p_value'])}]{_fmt_p(row['p_value'])}{_sig_stars(row['p_value'])}[/]",
        )
    return tbl


def _kw_table(kw: dict) -> Table:
    heads = (f"[dim]{thing}[/]" for thing in ("dimension", "H", "p"))
    tbl = Table(*heads, box=None, padding=(0, 2))
    for dim, vals in kw.items():
        tbl.add_row(
            dim,
            f"[cyan]{vals['H']:.2f}[/]",
            f"[{_p_color(vals['p_value'])}]{_fmt_p(vals['p_value'])}[/]",
        )
    return tbl


def _syntax_table(syn_rho: pd.DataFrame) -> Table:
    syn_arr = syn_rho.to_numpy()
    col_names = [c.removeprefix("ruff_") for c in syn_rho.columns]
    tbl = Table("", *col_names, box=None, padding=(0, 2))
    for i, cdim in enumerate(CODE_DIMS):
        cells = [f"[{_rho_color(syn_arr[i, j])}]{syn_arr[i, j]:.3f}[/]" for j in range(syn_arr.shape[1])]
        tbl.add_row(cdim, *cells)
    return tbl


######################### Save Results #########################


def _save(
    desc: dict,
    rho_df: pd.DataFrame,
    p_adj_df: pd.DataFrame,
    ranking: pd.DataFrame,
    reg: dict,
    model_corr: list[ModelCorr],
    kw: dict,
    syn_rho: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "descriptive": desc,
        "spearman_matrix": rho_df.to_dict(),
        "spearman_p_adj": p_adj_df.to_dict(),
        "prompt_dim_ranking": ranking.to_dict(orient="records"),
        "composite_regression": reg,
        "per_model_correlation": list(model_corr),
        "kruskal_wallis": kw,
        "syntax_vs_semantic_rho": syn_rho.to_dict(),
    }
    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    cout(f"\n[dim]Saved: {out_path}[/]")


def main() -> None:
    cout("Correlation Analysis :: [green]prompt2code-eval[/]\n")

    df = parquet_cache(CACHE_DIR / "analysis_input.parquet", _load_df, log=True)

    desc = descriptive_stats(df)
    rho_df, p_adj_df = spearman_matrix(df)
    ranking = prompt_dim_ranking(rho_df)
    reg = composite_regression(df)
    model_corr = per_model_correlation(df)
    kw = kruskal_wallis(df)
    syn_rho, _ = syntax_vs_semantic(df)

    n_models = len(df["model"].unique())
    cout(f"\nModel Distribution ({desc['parseable_rate']:.1%} parseable; {n_models:,} models):")
    cout(_model_count_table(desc))

    cout("\nSpearman rho (prompt -> code, Bonferroni-adjusted):")
    cout(
        "[dim]How strongly each prompt quality predicts each code quality dimension."
        " Values near 1 = strong positive link; stars = statistically significant.[/]"
    )
    cout(_matrix_table(rho_df, p_adj_df))
    cout("-> [dim]* p<0.05  ** p<0.01  *** p<0.001[/]")

    cout("\n[bold]Prompt dim ranking (mean rho across code dims)[/]:")
    cout(
        "[dim]Which prompt characteristic (clarity / specificity / completeness)"
        " best predicts overall code quality, on average.[/]"
    )
    cout(_ranking_table(ranking))

    cout("\n[bold]Composite regression[/]:")
    cout(
        "[dim]Treats all prompt scores as one combined number and fits a straight line"
        " to predict overall code quality.[/]"
    )
    cout(_regression_line(reg))

    cout("\n[bold]Per-model Spearman rho (composite prompt -> code)[/]:")
    cout("[dim]Same prompt->code correlation, split by which LLM generated the code.[/]")
    cout(_model_corr_table(model_corr))

    cout("\n[bold]Kruskal-Wallis H (code quality across models)[/]")
    cout(
        "[dim]Tests whether different LLMs produce code of meaningfully different quality"
        " -- not just correlated, but different averages.[/]"
    )
    cout(_kw_table(kw))

    cout("\n[bold]Syntax vs semantic rho (code dims vs static metrics)[/]:")
    cout(
        "[dim]Checks if the AI judge's scores agree with static analysis tools"
        " (ruff, complexity). Validates the scoring method.[/]"
    )
    cout(_syntax_table(syn_rho))

    _save(desc, rho_df, p_adj_df, ranking, reg, model_corr, kw, syn_rho)


if __name__ == "__main__":
    with graceful_exit("Analysis cancelled"):
        main()
