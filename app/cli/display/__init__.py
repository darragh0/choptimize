from __future__ import annotations

from typing import TYPE_CHECKING

from common.utils.console import cout
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from app.engine.models import AnalysisResult, Antipattern, ImprovementResult, SimilarPrompt, Technique


def _display_antipatterns(pats: list[Antipattern], *, verbose: bool) -> None:
    if pats:
        cout("\n[red]Detected Antipatterns[/]:")
        for ap in pats:
            cout(f"  [red]✗[/] {ap.name}: [dim]{ap.why_ineffective}[/]")
            cout(f"    [green]Instead:[/] {ap.instead}")
            if verbose:
                cout(f"    [dim]Evidence: {ap.evidence}[/]")
                cout(f"    [dim]Misconception: {ap.misconception}[/]")


def _display_techs(techs: list[Technique], *, verbose: bool) -> None:
    if techs:
        cout("\nSuggested Techniques:")
        for t in techs:
            dims = ", ".join(t.improves)
            cout(f"  [green]•[/] [bold]{t.name}[/] [dim](improves {dims})[/]")
            cout(f"    {t.description}")
            if verbose:
                cout(f"    [dim]Evidence: {t.evidence}[/]")


def _display_similar(sim_prompts: list[SimilarPrompt], *, verbose: bool) -> None:
    if sim_prompts and verbose:
        cout("\nSimilar Prompts from Dataset:")
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
    cout("\nPrompt Quality Scores:")
    for dim in ("clarity", "specificity", "completeness"):
        ds = getattr(result.scores, dim)
        cout(f"  [dim]{dim:<15}[/] {score_bar(ds.score)}")
        cout(f"  {' ' * 15} [dim]{ds.explanation}[/]")

    overall = result.scores.overall
    cout(f"\n  [bold]{'overall':<15}[/] {overall:.1f} / 5.0")

    if result.scores.summary:
        cout(f"\n{result.scores.summary}")

    _display_antipatterns(result.detected_antipatterns, verbose=verbose)
    _display_techs(result.relevant_techniques, verbose=verbose)
    _display_similar(result.similar_prompts, verbose=verbose)
    _display_imp(result.improvement, verbose=verbose)


__all__ = ["display_result", "score_bar"]
