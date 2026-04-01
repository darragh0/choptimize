import json
import os
import re

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

    def complete_json(self, system: str, user: str, *, retries: int = 2) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for attempt in range(retries + 1):
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                if match := re.search(r"\{.*}", raw, re.DOTALL):
                    try:
                        return json.loads(match.group())
                    except json.JSONDecodeError:
                        pass
                if attempt < retries:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": "That was not valid JSON. Respond with ONLY valid JSON, no other text."})
                    continue
                raise ValueError(f"LLM returned invalid JSON after {retries + 1} attempts: {raw[:200]}")
