from __future__ import annotations

from pydantic import BaseModel

from app.engine.types import ImprovesDim, Score, ScoreCont, TechniqueCat  # noqa: TC001


class Technique(BaseModel):
    name: str
    category: TechniqueCat
    when_to_use: str
    description: str
    evidence: str
    example: str
    improves: list[ImprovesDim]

    def one_liner(self) -> str:
        return f"{self.name}: {self.description}. {self.when_to_use}"


class Antipattern(BaseModel):
    name: str
    description: str
    why_ineffective: str
    evidence: str
    misconception: str
    instead: str


class TechniquesSchema(BaseModel):
    techniques: list[Technique]
    antipatterns: list[Antipattern]


class DimensionScore(BaseModel):
    score: Score
    explanation: str


class ScoreResult(BaseModel):
    clarity: DimensionScore
    specificity: DimensionScore
    completeness: DimensionScore
    summary: str
    code_quality_outlook: str = ""

    @property
    def overall(self) -> ScoreCont:
        return (self.clarity.score + self.specificity.score + self.completeness.score) / 3


class SimilarPrompt(BaseModel):
    prompt: str
    scores: dict[str, Score]
    distance: float


class ImprovementChange(BaseModel):
    dimension: str
    technique_applied: str
    explanation: str


class ImprovementResult(BaseModel):
    improved_prompt: str
    changes: list[ImprovementChange]


class AnalysisResult(BaseModel):
    scores: ScoreResult
    similar_prompts: list[SimilarPrompt]
    relevant_techniques: list[Technique]
    detected_antipatterns: list[Antipattern] = []
    improvement: ImprovementResult | None = None
