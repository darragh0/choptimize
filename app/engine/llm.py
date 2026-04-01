import json
import os

from openai import OpenAI

_DEFAULT_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "gemma3:27b"


class LLMClient:
    base_url: str | None
    model: str
    api_key: str | None
    _client: OpenAI

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.environ.get("CHOPTIMIZE_LLM_MODEL", _DEFAULT_MODEL)
        self._client = OpenAI(
            base_url=base_url or os.environ.get("CHOPTIMIZE_LLM_URL", _DEFAULT_URL),
            api_key=api_key or os.environ.get("CHOPTIMIZE_LLM_API_KEY", "ollama"),
        )

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def complete_json(self, system: str, user: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")
