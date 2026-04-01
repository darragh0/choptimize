"""Figures for prompt-quality vs code-quality analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from common.utils.console import cout
from matplotlib.patches import Patch

from analysis.common import sig_stars
from analysis.correlations import CODE_DIMS, OUTPUT_DIR, PROMPT_DIMS

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from matplotlib.figure import Figure

    from .correlations import ModelCorr


FIG_DIR: Path = OUTPUT_DIR / "figures"


def _setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
        }
    )


def _save(fig: Figure, name: str) -> None:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    cout(f"  [dim]Saved: {path}[/]")


def _label(dim: str) -> str:
    return dim.capitalize()


def correlation_heatmap(rho_df: pd.DataFrame, p_adj_df: pd.DataFrame) -> None:
    """Annotated heatmap of prompt-dim x code-dim Spearman correlations.

    This is the central figure: shows which prompt qualities predict which code qualities.
    """
    rho = rho_df.to_numpy()
    p = p_adj_df.to_numpy()

    annot = np.array(
        [[f"{rho[i, j]:.3f}{sig_stars(p[i, j])}" for j in range(rho.shape[1])] for i in range(rho.shape[0])]
    )

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(
        rho_df.rename(index=_label, columns=_label),
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
        ax=ax,
    )
    ax.set_xlabel("Code Quality Dimension")
    ax.set_ylabel("Prompt Quality Dimension")
    ax.set_title("Prompt–Code Quality Correlations (Spearman ρ, Bonferroni-adjusted)")
    _save(fig, "correlation_heatmap")


def score_distributions(df: pd.DataFrame) -> None:
    """Violin plots showing how each quality dimension is distributed across the dataset.

    Reveals ceiling/floor effects and whether the LLM judge used the full 1-5 range.
    """
    all_dims = [*PROMPT_DIMS, *CODE_DIMS]
    melted = df[all_dims].melt(var_name="Dimension", value_name="Score")
    melted["Dimension"] = melted["Dimension"].map(_label)
    prompt_labels = {"Clarity", "Specificity", "Completeness"}
    melted["Type"] = melted["Dimension"].apply(lambda d: "Prompt" if d in prompt_labels else "Code")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, squeeze=False)
    ax1, ax2 = axes[0, 0], axes[0, 1]

    prompt_data = melted[melted["Type"] == "Prompt"]
    sns.violinplot(data=prompt_data, x="Dimension", y="Score", ax=ax1, inner="quartile", cut=0)
    ax1.set_title("Prompt Quality Dimensions")
    ax1.set_xlabel("")
    ax1.set_ylim(0.5, 5.5)
    ax1.set_yticks(range(1, 6))

    code_data = melted[melted["Type"] == "Code"]
    sns.violinplot(data=code_data, x="Dimension", y="Score", ax=ax2, inner="quartile", cut=0)
    ax2.set_title("Code Quality Dimensions")
    ax2.set_xlabel("")

    fig.suptitle(f"Score Distributions Across Quality Dimensions (n = {len(df):,})")
    fig.tight_layout()
    _save(fig, "score_distributions")


def composite_scatter(df: pd.DataFrame, reg: dict[str, float]) -> None:
    """Scatter plot of mean prompt quality vs mean code quality with regression line.

    Shows the overall relationship between the two composite measures.
    With 26K points, we use a hexbin to avoid overplotting.
    """
    prompt_q = df[PROMPT_DIMS].mean(axis=1)
    code_q = df[CODE_DIMS].mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(prompt_q, code_q, gridsize=30, cmap="YlGnBu", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Count")

    # Regression line
    xs = np.linspace(float(prompt_q.min()), float(prompt_q.max()), 100)
    ys = reg["slope"] * xs + reg["intercept"]
    ax.plot(
        xs,
        ys,
        color="red",
        linewidth=2,
        label=(
            f"OLS: y = {reg['slope']:.3f}x + {reg['intercept']:.3f}\n"
            f"R² = {reg['r_squared']:.3f}, ρ = {reg['spearman_rho']:.3f}"
        ),
    )

    ax.set_xlabel("Mean Prompt Quality")
    ax.set_ylabel("Mean Code Quality")
    ax.set_title("Composite Prompt Quality vs Code Quality")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "composite_scatter")


def per_model_bars(model_corr: list[ModelCorr]) -> None:
    """Horizontal bar chart of Spearman rho per model.

    Shows whether the prompt-code relationship is consistent across different LLMs.
    """
    if not model_corr:
        return

    models = [m["model"] for m in model_corr]
    rhos = [m["rho"] for m in model_corr]
    ns = [m["n"] for m in model_corr]
    sig = [m["p_value"] < 0.05 for m in model_corr]

    fig, ax = plt.subplots(figsize=(8, max(3, len(models) * 0.4)))
    colors = ["#2ecc71" if s else "#95a5a6" for s in sig]
    bars = ax.barh(models, rhos, color=colors, edgecolor="white", linewidth=0.5)

    for bar, n in zip(bars, ns, strict=True):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"n={n:,}",
            va="center",
            fontsize=8,
            color="dimgray",
        )

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman ρ (composite prompt → code quality)")
    ax.set_title("Prompt–Code Correlation by Model")
    ax.invert_yaxis()

    ax.legend(
        handles=[Patch(color="#2ecc71", label="p < 0.05"), Patch(color="#95a5a6", label="Not significant")],
        loc="lower right",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, "per_model_correlation")


def syntax_validation_heatmap(syn_rho: pd.DataFrame) -> None:
    """Heatmap of code-dim vs syntax-metric correlations.

    Validates the LLM judge: semantic scores should agree with objective static metrics
    (e.g., readability should correlate with maintainability index).
    """
    labels = {
        "ruff_errors": "Errors",
        "ruff_warnings": "Warnings",
        "ruff_flake8": "Flake8",
        "ruff_bugbear": "Bugbear",
        "ruff_security": "Security",
        "complexity": "Complexity",
        "maintainability": "Maintainability",
    }

    fig, ax = plt.subplots(figsize=(8, 3.5))
    display_df = syn_rho.rename(index=_label, columns=lambda c: labels.get(c, c))
    sns.heatmap(
        display_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
        ax=ax,
    )
    ax.set_xlabel("Static Analysis Metric")
    ax.set_ylabel("LLM-Judged Code Dimension")
    ax.set_title("Semantic Score vs Static Metric Agreement")
    _save(fig, "syntax_validation")


def generate_all(
    df: pd.DataFrame,
    rho_df: pd.DataFrame,
    p_adj_df: pd.DataFrame,
    reg: dict[str, float],
    model_corr: list[ModelCorr],
    syn_rho: pd.DataFrame,
) -> None:
    """Generate all figures for the report."""
    _setup()
    cout("\nGenerating figures:")
    correlation_heatmap(rho_df, p_adj_df)
    score_distributions(df)
    composite_scatter(df, reg)
    per_model_bars(model_corr)
    syntax_validation_heatmap(syn_rho)
    cout(f"\n[green]All figures saved to {FIG_DIR}[/]")
