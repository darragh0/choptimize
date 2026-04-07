from __future__ import annotations

from typing import TYPE_CHECKING

from rich.box import MINIMAL
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from common.utils.console import cout

_P_DIMS = {"clarity", "specificity", "completeness"}
_C_DIMS = {"correctness", "robustness", "readability", "efficiency"}

if TYPE_CHECKING:
    from app.engine.models import AnalysisResult, Antipattern, ImprovementResult, SimilarPrompt, Technique


def _fmt_antipatterns(pats: list[Antipattern]) -> str:
    lines: list[str] = []
    for i, ap in enumerate(pats):
        lines.append(f"[red]✗[/] {ap.name}")
        lines.append(f"  [dim]{ap.why_ineffective}[/]")
        lines.append(f"[green]→[/] {ap.instead}")
        if i < len(pats) - 1:
            lines.append("")
    return "\n".join(lines)


def _display_outlook_and_antipatterns(outlook: str, pats: list[Antipattern]) -> None:
    has_outlook = bool(outlook)
    has_pats = bool(pats)
    if not has_outlook and not has_pats:
        return
    cout()
    if has_outlook and has_pats:
        tbl = Table(expand=True, show_edge=False, show_footer=False, pad_edge=False, padding=(0, 2), box=MINIMAL)
        tbl.add_column("[cyan]Code Quality Outlook[/]", ratio=1)
        tbl.add_column("[red]Antipatterns Detected[/]", ratio=1)
        tbl.add_row(outlook, _fmt_antipatterns(pats))
        cout(tbl)
    elif has_outlook:
        cout(Rule("[cyan]Code Quality Outlook[/]", style="dim"))
        cout(Padding(outlook, pad=(0, 2)))
    else:
        cout(Rule("[red]Antipatterns Detected[/]", style="dim"))
        cout(Padding(_fmt_antipatterns(pats), pad=(0, 2)))


def _display_techs(techs: list[Technique]) -> None:
    if not techs:
        return
    cout()
    cout("[green]Suggested Techniques[/]")
    for i, t in enumerate(techs):
        dims = ", ".join(t.improves)
        cout(f"    [green]•[/] {t.name} [dim]({dims})[/]")
        cout(Padding(f"[dim]{t.description}[/]", pad=(0, 0, 0, 6)))
        if i < len(techs) - 1:
            cout()


def _display_similar(sim_prompts: list[SimilarPrompt]) -> None:
    if not sim_prompts:
        return
    cout()
    cout("[magenta]Similar Prompts[/]")
    for i, sp in enumerate(sim_prompts):
        p_scores = ", ".join(f"{k}={v}" for k, v in sp.scores.items() if k in _P_DIMS)
        c_scores = ", ".join(f"{k}={v}" for k, v in sp.scores.items() if k in _C_DIMS)
        cout(f"    [dim]prompt: ({p_scores})[/]")
        if c_scores:
            cout(f"    [dim]  code: ({c_scores})[/]")
        cout(Padding(f"{sp.prompt[:120]}{'...' if len(sp.prompt) > 120 else ''}", pad=(0, 0, 0, 4)))
        if i < len(sim_prompts) - 1:
            cout()


def _display_imp(imp: ImprovementResult | None) -> None:
    if imp is None:
        return
    cout()
    cout(
        Panel(
            Text(imp.improved_prompt),
            title="[green]Improved Prompt[/]",
            border_style="green",
        )
    )
    for ch in imp.changes:
        cout(f"    [green]↳ {ch.dimension}:[/] applied [cyan]{ch.technique_applied}[/]")
        expl = ch.explanation if len(ch.explanation) <= 120 else ch.explanation[:117] + "..."
        cout(Padding(f"[dim]{expl}[/]", pad=(0, 0, 0, 6)))
    cout()


def _score_bar(score: int, max_score: int = 5) -> str:
    colors = {1: "red", 2: "orange1", 3: "yellow3", 4: "green", 5: "green"}
    filled = "█" * score
    empty = "░" * (max_score - score)
    return f"[{colors.get(score, 'white')}]{filled}[/][dim]{empty}[/] {score}/{max_score}"


def _overall_color(score: float) -> str:
    if score >= 4:
        return "green"
    if score >= 3:
        return "yellow3"
    if score >= 2:
        return "orange1"
    return "red"


def display_result(result: AnalysisResult) -> None:
    s = result.scores

    # --- Summary verdict ---
    if s.summary:
        cout()
        cout(Panel(s.summary, border_style="dim", padding=(0, 1)))

    # --- Score cards ---
    cout()
    cout(f"  overall          [{_overall_color(s.overall)}]{s.overall:.1f}[/] [dim]/ 5[/]")
    cout()
    for dim in ("clarity", "specificity", "completeness"):
        ds = getattr(s, dim)
        cout(f"  {dim:<15} {_score_bar(ds.score)}")
        cout(f"                    [dim]{ds.explanation}[/]")
        cout()

    # --- Code quality outlook & antipatterns (side by side) ---
    _display_outlook_and_antipatterns(s.code_quality_outlook or "", result.detected_antipatterns)

    # --- Sections ---
    _display_techs(result.relevant_techniques)
    _display_similar(result.similar_prompts)
    _display_imp(result.improvement)


__all__ = ["display_result"]
