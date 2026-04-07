from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from app.engine.improver import improve_prompt
from app.engine.llm import LLMClient
from app.engine.models import AnalysisResult
from app.engine.retriever import Retriever
from app.engine.scorer import score_prompt
from app.engine.validator import validate

if TYPE_CHECKING:
    from app.engine.types import LLMService


class Engine:
    _llm: LLMClient
    _retriever: Retriever | None

    def __init__(
        self,
        service: LLMService | None = None,
        model: str | None = None,
        llm_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._llm = LLMClient(service=service or "ollama", base_url=llm_url, model=model, api_key=api_key)
        self._retriever = None

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever

    def analyze(self, prompt: str, *, improve: bool = False, show_raw: bool) -> AnalysisResult:
        if msg := validate(prompt):
            raise ValueError(msg)

        # Retrieval first, results feed into scorer
        with ThreadPoolExecutor(max_workers=3) as pool:
            similar_fut = pool.submit(self.retriever.find_similar_prompts, prompt)
            techniques_fut = pool.submit(self.retriever.find_techniques, prompt)
            antipattern_fut = pool.submit(self.retriever.detect_antipatterns, prompt)

            similar = similar_fut.result()
            techniques = techniques_fut.result()
            antipatterns = antipattern_fut.result()

        # Score with RAG context for grounded feedback
        scores = score_prompt(
            self._llm,
            prompt,
            techniques=techniques,
            similar=similar,
            antipatterns=antipatterns,
            show_raw=show_raw,
        )

        improvement = None
        if improve:
            weak_dims = [dim for dim in ("clarity", "specificity", "completeness") if getattr(scores, dim).score <= 3]
            weak_techniques = self.retriever.find_techniques(prompt, weak_dims)
            improvement = improve_prompt(self._llm, prompt, scores, weak_techniques, similar, show_raw=show_raw)

        return AnalysisResult(
            scores=scores,
            similar_prompts=similar,
            relevant_techniques=techniques,
            detected_antipatterns=antipatterns,
            improvement=improvement,
        )
