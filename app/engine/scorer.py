from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from app.engine.models import ScoreResult
from app.engine.types import CODE_DIMS, PROMPT_DIMS  # noqa: TC001

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
        "relevant prompt engineering techniques, and detected antipatterns. "
        "Each similar prompt includes both prompt quality scores AND code quality outcomes "
        "(correctness, robustness, readability, efficiency) showing what code quality "
        "resulted from that prompt."
    )
    parts.append(
        "\nFor each dimension's explanation: write ONE concise sentence (under 20 words) "
        "identifying the key issue or strength. Do not elaborate or suggest fixes — the summary handles that."
    )
    parts.append(
        "\nFor the summary: give 2-3 concrete, actionable tips to improve the prompt. "
        "Don't just say what's wrong — say what to do about it."
    )
    parts.append(
        "\nFor code_quality_outlook: based on the code quality outcomes of the similar prompts, "
        "write 1-2 sentences predicting what code quality the user can expect from their prompt. "
        "Reference specific dimensions (e.g. 'similar prompts produced code with low correctness'). "
        "If no similar prompts are available, leave this field empty."
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
            p_scores = ", ".join(f"{k}={v}" for k, v in s.scores.items() if k in PROMPT_DIMS)
            c_scores = ", ".join(f"{k}={v}" for k, v in s.scores.items() if k in CODE_DIMS)
            sim_lines.append(f"- [prompt: {p_scores} | code: {c_scores}]: {s.prompt[:200]}")

        code_avgs: dict[str, float] = {}
        for dim in CODE_DIMS:
            vals = [s.scores[dim] for s in similar if dim in s.scores]
            if vals:
                code_avgs[dim] = sum(vals) / len(vals)
        avg_str = ", ".join(f"{k}={v:.1f}" for k, v in code_avgs.items())

        parts.append(f"<SIMILAR_SCORED_PROMPTS>\n{chr(10).join(sim_lines)}\n</SIMILAR_SCORED_PROMPTS>")
        if avg_str:
            parts.append(
                "<CODE_QUALITY_OUTLOOK_DATA>\n"
                f"Average code quality from similar prompts: {avg_str}\n"
                "</CODE_QUALITY_OUTLOOK_DATA>"
            )

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
