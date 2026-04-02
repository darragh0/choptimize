from __future__ import annotations

from typing import TYPE_CHECKING, Final

from app.engine.models import ImprovementResult

if TYPE_CHECKING:
    from app.engine.llm import LLMClient
    from app.engine.models import ScoreResult, SimilarPrompt, Technique


_SYSTEM: Final = """\
<task>
Your task is to optimize a developer's prompt using grounded evidence. \
You will be given a piece of text containing the following inputs, each of which \
are wrapped in their corresponding XML-style tags:

<inputs>
1. ORIGINAL_PROMPT — A developer's original prompt.
2. QUALITY_SCORES — Quality scores on clarity, specificity, and completeness (1-5).
3. RELEVANT_TECHNIQUES — Relevant prompt engineering techniques with evidence.
4. SIMILAR_PROMPTS_FROM_DATASET — Similar prompts from a dataset with their scores.
</inputs>

<instructions>
Given the inputs, rewrite the prompt to improve weak dimensions. \
Apply the provided techniques. \
Explain each change you made and which technique you applied.

Respond with ONLY valid JSON matching the provided schema.
</instructions>
</task>\
"""


def _build_context(
    prompt: str,
    scores: ScoreResult,
    techniques: list[Technique],
    similar: list[SimilarPrompt],
) -> str:
    parts = [f"<ORIGINAL_PROMPT>{prompt}</ORIGINAL_PROMPT>"]
    simprompt_cutoff = 200

    parts.append(
        "<QUALITY_SCORES>"
        f"- clarity: {scores.clarity.score}/5 — {scores.clarity.explanation}\n"
        f"- specificity: {scores.specificity.score}/5 — {scores.specificity.explanation}\n"
        f"- completeness: {scores.completeness.score}/5 — {scores.completeness.explanation}"
        "</QUALITY_SCORES>"
    )

    if techniques:
        tech_lines = []
        tech_lines.extend(f"- **{t.name}** ({t.evidence}): {t.description}" for t in techniques)
        parts.append(f"<RELEVANT_TECHNIQUES>{'\n'.join(tech_lines)}</RELEVANT_TECHNIQUES>")

    if similar:
        sim_lines = []
        for s in similar:
            scores_str = ", ".join(f"{k}={v}" for k, v in s.scores.items())
            sim_lines.append(f"- [{scores_str}]: {s.prompt[:simprompt_cutoff]}")
        parts.append(f"<SIMILAR_PROPMTS_FROM_DATASET>{'\n'.join(sim_lines)}</SIMILAR_PROMPTS_FROM_DATASET>")

    return "\n".join(parts)


def improve_prompt(
    client: LLMClient,
    prompt: str,
    scores: ScoreResult,
    techniques: list[Technique],
    similar: list[SimilarPrompt],
    *,
    show_raw: bool,
) -> ImprovementResult:
    context = _build_context(prompt, scores, techniques, similar)
    return client.complete_json(_SYSTEM, context, response_model=ImprovementResult, show_raw=show_raw)
