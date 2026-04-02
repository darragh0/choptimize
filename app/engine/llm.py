import json
import os
from typing import Any

from common.utils.console import cout
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
from pydantic import BaseModel

_DEFAULT_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "gemma3:27b"


class _Conversation:
    messages: list[ChatCompletionMessageParam]

    def __init__(self, system: str, user: str) -> None:
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def usr(self, msg: str) -> None:
        self.messages.append({"role": "user", "content": msg})

    def llm(self, msg: str) -> None:
        self.messages.append({"role": "assistant", "content": msg})


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

    def config(self, response_model: type[BaseModel]) -> dict[str, Any]:
        return {
            "model": self.model,
            "response_format": ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=JSONSchema(
                    name=response_model.__name__,
                    schema=response_model.model_json_schema(),
                    strict=True,
                ),
            ),
        }

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        response_model: type[BaseModel],
        retries: int = 2,
        show_raw: bool = False,
    ) -> dict[str, Any]:
        conv = _Conversation(system, user)
        config = self.config(response_model)

        for attempt in range(retries + 1):
            raw = (
                self._client.chat.completions.create(**config, messages=conv.messages).choices[0].message.content or ""
            )
            if show_raw:
                cout(f"Raw LLM response ({attempt}):")
                cout(f"{raw}\n")

            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                if attempt < retries:
                    conv.llm(raw)
                    conv.usr("That was not valid JSON. Respond with ONLY valid JSON, no other text.")
                    continue
                msg = f"LLM returned invalid JSON after {retries + 1} attempts: {raw[:200]}"
                raise ValueError(msg) from e

        raise ValueError("unreachable")
