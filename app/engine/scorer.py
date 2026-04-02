from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from app.engine.models import DimensionScore, ScoreResult

if TYPE_CHECKING:
    from app.engine.llm import LLMClient
    from app.engine.models import Antipattern, SimilarPrompt, Technique

_RUBRIC = json.loads((Path(__file__).parent / "knowledge" / "rubric.json").read_text())


def _build_system_prompt() -> str:
    parts = [
        "You are an expert prompt quality evaluator for coding prompts.",
        _RUBRIC["context"],
        "\nScore the given developer prompt on the following dimensions (1-5 scale):\n",
    ]

    for dim, info in _RUBRIC["dimensions"].items():
        parts.append(f"<{dim.upper()}>")
        parts.append(info["description"])
        for level, desc in info["levels"].items():
            parts.append(f"  {level}: {desc}")
        parts.append(f"</{dim.upper()}>")
        parts.append("")

    parts.append(_RUBRIC["scoring_instructions"])
    parts.append(
        "\nYou will receive the prompt to evaluate along with grounding context: "
        "similar prompts from a scored dataset (use as calibration anchors), "
        "relevant prompt engineering techniques, and detected antipatterns."
    )
    parts.append(
        "\nFor each dimension's explanation: reference specific aspects of the prompt, "
        "mention relevant techniques the user could apply, and flag any antipatterns detected."
    )
    parts.append(
        "\nFor the summary: give 2-3 concrete, actionable tips to improve the prompt. "
        "Don't just say what's wrong — say what to do about it."
    )
    parts.append("\nRespond with ONLY valid JSON matching the provided schema.")

    return "\n".join(parts)


_SYSTEM = _build_system_prompt()


def _build_user_message(
    prompt: str,
    techniques: list[Technique],
    similar: list[SimilarPrompt],
    antipatterns: list[Antipattern],
) -> str:
    parts = [f"<PROMPT_TO_EVALUATE>\n{prompt}\n</PROMPT_TO_EVALUATE>"]

    if similar:
        sim_lines = []
        for s in similar:
            scores_str = ", ".join(f"{k}={v}" for k, v in s.scores.items())
            sim_lines.append(f"- [{scores_str}]: {s.prompt[:200]}")
        parts.append(f"<SIMILAR_SCORED_PROMPTS>\n{chr(10).join(sim_lines)}\n</SIMILAR_SCORED_PROMPTS>")

    if techniques:
        tech_lines = [f"- {t.name}: {t.description}" for t in techniques]
        parts.append(f"<RELEVANT_TECHNIQUES>\n{chr(10).join(tech_lines)}\n</RELEVANT_TECHNIQUES>")

    if antipatterns:
        ap_lines = [f"- {ap.name}: {ap.why_ineffective}" for ap in antipatterns]
        parts.append(f"<DETECTED_ANTIPATTERNS>\n{chr(10).join(ap_lines)}\n</DETECTED_ANTIPATTERNS>")

    return "\n\n".join(parts)


def score_prompt(
    client: LLMClient,
    prompt: str,
    *,
    techniques: list[Technique],
    similar: list[SimilarPrompt],
    antipatterns: list[Antipattern],
    show_raw: bool,
) -> ScoreResult:
    user_msg = _build_user_message(prompt, techniques, similar, antipatterns)
    return client.complete_json(_SYSTEM, user_msg, response_model=ScoreResult, show_raw=show_raw)
    return ScoreResult(
        clarity=DimensionScore(**raw["clarity"]),
        specificity=DimensionScore(**raw["specificity"]),
        completeness=DimensionScore(**raw["completeness"]),
        summary=raw.get("summary", ""),
    )
