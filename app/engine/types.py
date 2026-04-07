from typing import Annotated, Final, Literal, TypedDict

from annotated_types import Ge, Le

type Score = Annotated[int, Ge(1), Le(5)]
type ScoreCont = Annotated[float, Ge(1), Le(5)]

type LLMService = Literal["ollama", "openai", "gemini"]

PROMPT_DIMS: Final = ("clarity", "specificity", "completeness")
CODE_DIMS: Final = ("correctness", "robustness", "readability", "efficiency")

type TechniqueCat = Literal[
    "example-driven",
    "structure",
    "context",
    "constraint",
    "reasoning",
    "verification",
    "efficiency",
    "completeness",
]

type ImprovesDim = Literal["clarity", "specificity", "completeness"]


class DSRowMeta(TypedDict, total=False):
    model: str
    clarity: Score
    specificity: Score
    completeness: Score
    correctness: Score
    robustness: Score
    readability: Score
    efficiency: Score
