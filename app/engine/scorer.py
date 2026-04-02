from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from app.engine.models import DimensionScore, ScoreResult

if TYPE_CHECKING:
    from app.engine.llm import LLMClient

_RUBRIC = json.loads((Path(__file__).parent / "knowledge" / "rubric.json").read_text())


def _build_system_prompt() -> str:
    parts = [
        "Your only task is to expertly evaluate the quality of a coding-prompt.",
        _RUBRIC["context"],
        "Score the given developer prompt on the following dimensions (1-5 scale):\n",
    ]

    for dim, info in _RUBRIC["dimensions"].items():
        parts.append(f"<{dim.upper()}>")
        parts.append(info["description"])
        for level, desc in info["levels"].items():
            parts.append(f"  {level}: {desc}")
        parts.append(f"</{dim.upper()}>")
        parts.append("")

    parts.append(f"{_RUBRIC['scoring_instructions']}")
    parts.append("\nRespond with ONLY valid JSON matching the provided schema.")

    return "\n".join(parts)


_SYSTEM = _build_system_prompt()


def score_prompt(client: LLMClient, prompt: str, *, show_raw: bool) -> ScoreResult:
    raw = client.complete_json(_SYSTEM, prompt, response_model=ScoreResult, show_raw=show_raw)
    return ScoreResult(
        clarity=DimensionScore(**raw["clarity"]),
        specificity=DimensionScore(**raw["specificity"]),
        completeness=DimensionScore(**raw["completeness"]),
        summary=raw.get("summary", ""),
    )
