from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app.engine.improver import improve_prompt
from app.engine.llm import LLMClient
from app.engine.models import AnalysisResult
from app.engine.retriever import Retriever
from app.engine.scorer import score_prompt


class Engine:
    _llm: LLMClient
    _retriever: Retriever | None

    def __init__(
        self,
        model: str | None = None,
        llm_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._llm = LLMClient(base_url=llm_url, model=model, api_key=api_key)
        self._retriever = None

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever

    def analyze(self, prompt: str, *, improve: bool = False, show_raw: bool) -> AnalysisResult:
        with ThreadPoolExecutor(max_workers=3) as pool:
            score_fut = pool.submit(score_prompt, self._llm, prompt, show_raw=show_raw)
            similar_fut = pool.submit(self.retriever.find_similar_prompts, prompt)
            antipattern_fut = pool.submit(self.retriever.detect_antipatterns, prompt)

            scores = score_fut.result()
            similar = similar_fut.result()
            antipatterns = antipattern_fut.result()

        weak_dims = [dim for dim in ("clarity", "specificity", "completeness") if getattr(scores, dim).score <= 3]
        techniques = self.retriever.find_techniques(prompt, weak_dims)

        improvement = None
        if improve:
            improvement = improve_prompt(self._llm, prompt, scores, techniques, similar)

        return AnalysisResult(
            scores=scores,
            similar_prompts=similar,
            relevant_techniques=techniques,
            detected_antipatterns=antipatterns,
            improvement=improvement,
        )
