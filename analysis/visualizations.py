"""Visualization plots for prompt vs. code quality analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from analysis.correlations import CODE_DIMS, PROMPT_DIMS

mpl.use("Agg")

_LOW_THRESHOLD = 2.5
_HIGH_THRESHOLD = 3.5

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import DataFrame


def plot_correlation_heatmap(df: DataFrame, output_dir: Path) -> None:
    """Heatmap of Pearson r between prompt and code dimensions."""
    matrix = np.zeros((len(PROMPT_DIMS), len(CODE_DIMS)))
    for i, pdim in enumerate(PROMPT_DIMS):
        for j, cdim in enumerate(CODE_DIMS):
            r, _ = stats.pearsonr(df[pdim], df[cdim])
            matrix[i, j] = r

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        matrix,
        xticklabels=[d.capitalize() for d in CODE_DIMS],
        yticklabels=[d.capitalize() for d in PROMPT_DIMS],
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        ax=ax,
    )
    ax.set_title("Prompt Quality vs. Code Quality — Pearson Correlations")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_scatter(df: DataFrame, output_dir: Path) -> None:
    """Scatter plot of composite prompt quality vs. composite code quality."""
    prompt_score = df[list(PROMPT_DIMS)].mean(axis=1)
    code_score = df[list(CODE_DIMS)].mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(prompt_score, code_score, alpha=0.05, s=8, color="steelblue")

    slope, intercept, r, _, _ = stats.linregress(prompt_score, code_score)
    x_line = np.linspace(prompt_score.min(), prompt_score.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=2, label=f"r²={r**2:.3f}")

    ax.set_xlabel("Composite Prompt Quality (mean of 3 dims)")
    ax.set_ylabel("Composite Code Quality (mean of 4 dims)")
    ax.set_title("Prompt Quality vs. Code Quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_prompt_vs_code.png", dpi=150)
    plt.close(fig)


def plot_quality_tiers(df: DataFrame, output_dir: Path) -> None:
    """Box plots of code quality by prompt quality tier."""
    prompt_score = df[list(PROMPT_DIMS)].mean(axis=1)
    tiers = prompt_score.apply(
        lambda x: "Low (1-2)" if x < _LOW_THRESHOLD else ("Mid (2.5-3.5)" if x < _HIGH_THRESHOLD else "High (3.5-5)")
    )
    df = df.copy()
    df["prompt_tier"] = tiers
    df["code_quality"] = df[list(CODE_DIMS)].mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    order = ["Low (1-2)", "Mid (2.5-3.5)", "High (3.5-5)"]
    sns.boxplot(data=df, x="prompt_tier", y="code_quality", order=order, ax=ax, palette="RdYlGn")
    ax.set_xlabel("Prompt Quality Tier")
    ax.set_ylabel("Composite Code Quality")
    ax.set_title("Code Quality Distribution by Prompt Quality Tier")
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot_quality_tiers.png", dpi=150)
    plt.close(fig)


def plot_dimension_distributions(df: DataFrame, output_dir: Path) -> None:
    """Bar charts of score distributions for each dimension."""
    all_dims = list(PROMPT_DIMS) + list(CODE_DIMS)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()

    for i, dim in enumerate(all_dims):
        ax = axes_flat[i]
        counts = df[dim].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color="steelblue")
        ax.set_title(dim.capitalize())
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_xticks(range(1, 6))

    # Hide unused subplot (7 dims, 8 slots)
    axes_flat[-1].set_visible(False)

    fig.suptitle("Score Distributions by Dimension", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "dimension_distributions.png", dpi=150)
    plt.close(fig)
