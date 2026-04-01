from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from common.utils.dataset import load_ds
from preproc.utils.progress import tracked

from app.engine.models import Antipattern, SimilarPrompt, Technique, TechniquesSchema

if TYPE_CHECKING:
    from chromadb import Collection
    from chromadb.api import ClientAPI
    from chromadb.api.types import Embeddable, EmbeddingFunction, Metadata
    from datasets import Dataset

    from app.engine.types import DSRowMeta


_CHROMA_DIR: Final = Path("data/chroma")
_TECHNIQUES_PATH: Final = Path(__file__).parent / "knowledge" / "techniques.json"
_EMBED_MODEL: Final = "all-MiniLM-L6-v2"

_DS_NAME: Final = "darragh0/prompt2code-eval"
_DS_REVISION: Final = "1c01b1b582c8d929f24fc05a13e108ee31de8a0d"

PROMPT_DIMS: Final = ("clarity", "specificity", "completeness")
CODE_DIMS: Final = ("correctness", "robustness", "readability", "efficiency")


_ANTIPATTERN_SIGNALS: Final[dict[str, list[str]]] = {
    "Role Prompting for Code Quality": [
        r"\bact as\b",
        r"\byou are a(?:n)?\s+(?:expert|senior|experienced)",
        r"\bpretend you(?:'re| are)\b",
        r"\bassume the role\b",
    ],
    "Politeness/Emotional Appeals": [
        r"\bplease\b.*\bcareful\b",
        r"\bmy job depends\b",
        r"\bthis is (?:very )?important\b",
        r"\bdo your best\b",
    ],
    "Vague Quality Adjectives": [
        r"\b(?:write|generate|create)\b.*\b(?:good|clean|best|high[- ]quality|production[- ]ready)\b",
    ],
    "Chain-of-Thought for Simple Code Tasks": [],  # Detected by scorer context, not regex
    "Excessive Prompt Length": [],  # Detected by length check
}


class Retriever:
    _client: ClientAPI
    _prompts: Collection
    _techniques: Collection
    _kbase: TechniquesSchema
    _antipatterns: list[Antipattern]
    _antipatterns_by_name: dict[str, Antipattern]

    def __init__(self, chroma_dir: Path = _CHROMA_DIR) -> None:
        ef = cast(
            "EmbeddingFunction[Embeddable]",
            SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL),
        )
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._prompts = self._client.get_or_create_collection("dataset_prompts", embedding_function=ef)
        self._techniques = self._client.get_or_create_collection("techniques", embedding_function=ef)
        self._kbase = TechniquesSchema.model_validate_json(_TECHNIQUES_PATH.read_text())
        self._antipatterns = self._kbase.antipatterns
        self._antipatterns_by_name = {ap.name: ap for ap in self._antipatterns}
        self._ensure_ingested()

    def _ensure_ingested(self) -> None:
        if self._techniques.count() == 0:
            self._ingest_techniques()
        if self._prompts.count() == 0:
            self._ingest_dataset()

    def _ingest_techniques(self) -> None:
        techs = self._kbase.techniques
        self._techniques.add(
            ids=[f"tech_{i}" for i in range(len(techs))],
            documents=[tech.one_liner() for tech in techs],
            metadatas=[t.model_dump(mode="json") for t in techs],
        )

    def _ingest_dataset(self) -> None:
        ds: Dataset = load_ds(
            _DS_NAME,
            revision=_DS_REVISION,
            split="test",
            overview=False,
            status=None,
        )
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[Metadata] = []

        for i, row in enumerate(ds):
            row = cast("dict[str, object]", row)
            prompt = str(row.get("prompt", ""))
            if not prompt:
                continue

            meta: DSRowMeta = {}
            for dim in (*PROMPT_DIMS, *CODE_DIMS):
                val = row.get(dim)
                if val is not None:
                    meta[dim] = int(cast("int", val))

            if "model" in row:
                meta["model"] = str(row["model"])

            ids.append(f"prompt_{i}")
            documents.append(prompt)
            metadatas.append(cast("Metadata", meta))

        if not ids:
            return

        batch_size = 50
        batches = [
            (ids[s : s + batch_size], documents[s : s + batch_size], metadatas[s : s + batch_size])
            for s in range(0, len(ids), batch_size)
        ]
        for _, (b_ids, b_docs, b_meta) in tracked(batches, "Indexing dataset", total=len(batches)):
            self._prompts.add(ids=b_ids, documents=b_docs, metadatas=b_meta)

    def find_similar_prompts(self, prompt: str, n: int = 3) -> list[SimilarPrompt]:
        results = self._prompts.query(query_texts=[prompt], n_results=n)

        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        distances = results.get("distances") or []
        if not any((documents, metadatas, distances)):
            return []

        out = []
        for doc, meta, dist in zip(documents[0], metadatas[0], distances[0], strict=False):
            scores = {
                k: int(v) for k, v in (meta or {}).items() if k in (*PROMPT_DIMS, *CODE_DIMS) and isinstance(v, int)
            }
            out.append(SimilarPrompt(prompt=str(doc), scores=scores, distance=float(dist)))
        return out

    def find_techniques(self, prompt: str, weak_dims: list[str] | None = None, n: int = 3) -> list[Technique]:
        query = prompt
        if weak_dims:
            query += f" (weak dimensions: {', '.join(weak_dims)})"

        results = self._techniques.query(query_texts=[query], n_results=n)
        metadatas = results.get("metadatas") or []
        if not metadatas:
            return []

        out = []
        for meta in metadatas[0]:
            if meta is None:
                continue
            raw = dict(meta)
            raw["improves"] = json.loads(str(raw["improves"]))
            out.append(Technique.model_validate(raw))

        return out

    def detect_antipatterns(self, prompt: str) -> list[Antipattern]:
        prompt_lower = prompt.lower()
        detected: list[Antipattern] = []

        for name, patterns in _ANTIPATTERN_SIGNALS.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    if ap := self._antipatterns_by_name.get(name):
                        detected.append(ap)
                    break

        if len(prompt.split()) > 500 and (ap := self._antipatterns_by_name.get("Excessive Prompt Length")):
            detected.append(ap)

        return detected
