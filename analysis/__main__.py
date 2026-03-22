#!/usr/bin/env python3

"""Entry point."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from analysis.visualizations import (
    plot_correlation_heatmap,
    plot_dimension_distributions,
    plot_quality_tiers,
    plot_scatter,
)
from common.utils.cache import graceful_exit
from common.utils.display import cout, pretty_path

if TYPE_CHECKING:
    from pathlib import Path


def _save_results(results: dict, output_dir: Path) -> None:
    results_path = output_dir / "analysis_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    cout(f"  Results saved to {pretty_path(results_path)}")


def run_analysis() -> None:
    """Run full statistical analysis pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cout("Loading data...")
    df = load_data()
    cout(f"  {len(df):,} rows loaded")

    cout("\nComputing descriptive statistics...")
    desc_stats = compute_descriptive_stats(df)
    for dim, s in desc_stats.items():
        cout(f"  [cyan]{dim:<16}[/] mean={s['mean']:.2f}  std={s['std']:.2f}  median={s['median']:.1f}")

    cout("\nComputing Pearson correlations...")
    correlations = compute_correlations(df)
    for pdim, cdims in correlations["prompt_vs_code"].items():
        for cdim, vals in cdims.items():
            sig = "*" if vals["p"] < P_THRESHOLD else ""
            cout(f"  {pdim} x {cdim}: r={vals['r']:+.4f} p={vals['p']:.6f}{sig}")

    cout("\nLinear regression (composite prompt -> code quality)...")
    regression = compute_regression(df)
    cout(f"  r²={regression['r_squared']:.4f}  slope={regression['slope']:.4f}  p={regression['p_value']:.6f}")

    cout("\nStratified analysis by model...")
    model_results = stratify_by_model(df)
    for model_name, vals in sorted(model_results.items(), key=lambda x: x[1]["r"], reverse=True):
        cout(f"  [cyan]{model_name:<30}[/] n={vals['n']:>5}  r={vals['r']:+.4f}")

    cout("\nGenerating visualizations...")
    plot_correlation_heatmap(df, OUTPUT_DIR)
    plot_scatter(df, OUTPUT_DIR)
    plot_quality_tiers(df, OUTPUT_DIR)
    plot_dimension_distributions(df, OUTPUT_DIR)
    cout(f"  Plots saved to {pretty_path(OUTPUT_DIR)}")

    _save_results(
        {
            "descriptive_stats": desc_stats,
            "correlations": correlations,
            "regression": regression,
            "model_stratified": model_results,
            "n_samples": len(df),
        },
        OUTPUT_DIR,
    )


def main() -> None:
    with graceful_exit("analysis stopped"):
        run_analysis()


if __name__ == "__main__":
    main()
