from __future__ import annotations

from typing import TYPE_CHECKING

from common.utils.console import cout
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from app.engine.models import AnalysisResult, Antipattern, ImprovementResult, SimilarPrompt, Technique


def _display_antipatterns(pats: list[Antipattern], *, verbose: bool) -> None:
    if pats:
        cout("\n[bold red]Detected Antipatterns[/]\n")
        for ap in pats:
            cout(f"  [red]✗[/] [bold]{ap.name}[/]")
            cout(f"    [dim]{ap.why_ineffective}[/]")
            cout(f"    [green]Instead:[/] {ap.instead}")
            if verbose:
                cout(f"    [dim]Evidence: {ap.evidence}[/]")


def _display_techs(techs: list[Technique], *, verbose: bool) -> None:
    if techs:
        cout("\n[bold]Suggested Techniques[/]\n")
        for t in techs:
            dims = ", ".join(t.improves)
            cout(f"  [green]•[/] [bold]{t.name}[/] [dim](improves {dims})[/]")
            if verbose:
                cout(f"    [dim]{t.description}[/]")
                cout(f"    [dim]Evidence: {t.evidence}[/]")


def _display_similar(sim_prompts: list[SimilarPrompt], *, verbose: bool) -> None:
    if sim_prompts and verbose:
        cout("\n[bold]Similar Prompts from Dataset[/]\n")
        for sp in sim_prompts:
            scores_str = ", ".join(
                f"{k}={v}" for k, v in sp.scores.items() if k in ("clarity", "specificity", "completeness")
            )
            cout(f"  [dim]({scores_str})[/]")
            cout(f"  {sp.prompt[:120]}{'…' if len(sp.prompt) > 120 else ''}\n")


def _display_imp(imp: ImprovementResult | None, *, verbose: bool) -> None:
    if imp is not None:
        cout()
        cout(
            Panel(
                Text(imp.improved_prompt),
                title="[bold green]Improved Prompt[/]",
                border_style="green",
            )
        )
        if verbose:
            for ch in imp.changes:
                cout(f"  [green]↳[/] [bold]{ch.dimension}[/]: applied [cyan]{ch.technique_applied}[/]")
                cout(f"    [dim]{ch.explanation}[/]")


def score_bar(score: int, max_score: int = 5) -> str:
    colors = {1: "red", 2: "red", 3: "yellow", 4: "green", 5: "bold green"}
    filled = "█" * score
    empty = "░" * (max_score - score)
    return f"[{colors.get(score, 'white')}]{filled}[/][dim]{empty}[/] {score}/{max_score}"


def display_result(result: AnalysisResult, *, verbose: bool) -> None:
    cout("\n[bold]Prompt Quality Scores[/]\n")
    for dim in ("clarity", "specificity", "completeness"):
        ds = getattr(result.scores, dim)
        cout(f"  {dim:<15} {score_bar(ds.score)}")
        if verbose:
            cout(f"  [dim]{' ' * 15} {ds.explanation}[/]")

    overall = result.scores.overall
    cout(f"\n  {'overall':<15} [bold]{overall:.1f}[/] / 5.0")

    _display_antipatterns(result.detected_antipatterns, verbose=verbose)
    _display_techs(result.relevant_techniques, verbose=verbose)
    _display_similar(result.similar_prompts, verbose=verbose)
    _display_imp(result.improvement, verbose=verbose)


__all__ = ["display_result", "score_bar"]
