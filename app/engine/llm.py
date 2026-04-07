from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

import openai
from common.utils.console import cerr, cout
from openai import OpenAI
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
from pydantic import BaseModel

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from app.engine.types import LLMService


class ServiceConfig(TypedDict):
    base_url: str
    api_key_env: str
    api_key_fallback: str
    default_model: str


SERVICE_DEFAULTS: dict[LLMService, ServiceConfig] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "CHOPTIMIZE_LLM_API_KEY",
        "api_key_fallback": "ollama",
        "default_model": "llama3.1:latest",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_fallback": "",
        "default_model": "gpt-4o",
    },
    "gemini": {
        "base_url": "",  # native SDK, not used
        "api_key_env": "GEMINI_API_KEY",
        "api_key_fallback": "",
        "default_model": "gemini-2.5-flash",
    },
}

_M = TypeVar("_M", bound=BaseModel)


def _resolve_key(service: LLMService, api_key: str | None = None) -> str:
    cfg = SERVICE_DEFAULTS[service]
    resolved = api_key or os.environ.get(cfg["api_key_env"]) or cfg["api_key_fallback"]
    if service != "ollama" and not resolved:
        env_var = cfg["api_key_env"]
        msg = f"{service} requires an API key (pass [cyan]--api-key[/] or set [cyan]{env_var}[/])"
        raise ValueError(msg)
    return resolved


def list_models(service: LLMService, api_key: str | None = None) -> list[str]:
    key = _resolve_key(service, api_key)

    if service == "gemini":
        from google import genai  # noqa: PLC0415

        try:
            client = genai.Client(api_key=key)
            raw = list(client.models.list())
        except Exception as e:  # noqa: BLE001
            msg = f"gemini: invalid API key or unreachable ({e})"
            raise ValueError(msg) from e
        return sorted(m.name.split("/")[-1] for m in raw if m.name and m.name.startswith("models/gemini"))

    client = OpenAI(
        base_url=SERVICE_DEFAULTS[service]["base_url"],
        api_key=key,
    )
    try:
        raw_models = client.models.list()
    except openai.AuthenticationError as e:
        msg = f"{service}: invalid API key"
        raise ValueError(msg) from e
    except Exception as e:  # noqa: BLE001
        msg = f"{service}: unreachable ({e})"
        raise ValueError(msg) from e

    ids = [m.id for m in raw_models]
    if service == "openai":
        ids = [m for m in ids if m.startswith(("gpt", "o", "chatgpt"))]
    return sorted(ids)


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
    service: LLMService
    model: str
    _client: OpenAI
    _gemini_client: Any

    def __init__(
        self,
        service: LLMService = "ollama",
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.service = service
        cfg = SERVICE_DEFAULTS[service]

        resolved_key = _resolve_key(service, api_key)
        resolved_url = base_url or os.environ.get("CHOPTIMIZE_LLM_URL") or cfg["base_url"]
        self.model = model or os.environ.get("CHOPTIMIZE_LLM_MODEL") or cfg["default_model"]

        if service == "gemini":
            from google import genai  # noqa: PLC0415

            self._gemini_client = genai.Client(api_key=resolved_key)
            self._client = None  # type: ignore[assignment]
        else:
            self._gemini_client = None
            self._client = OpenAI(base_url=resolved_url, api_key=resolved_key)

        self._validate_model(resolved_key)

    def _validate_model(self, api_key: str) -> None:
        if self.service == "gemini":
            try:
                raw = list(self._gemini_client.models.list())
            except Exception as e:  # noqa: BLE001
                msg = f"gemini: invalid API key or unreachable ({e})"
                raise ValueError(msg) from e
            available = sorted(m.name.split("/")[-1] for m in raw if m.name and m.name.startswith("models/gemini"))
        else:
            try:
                available = list_models(self.service, api_key)
            except ValueError as e:
                if "invalid API key" in str(e):
                    raise
                return

        if self.model not in available:
            shown = ", ".join(available[:10])
            suffix = " (showing first 10)" if len(available) > 10 else ""
            cerr()
            cerr(f"model [arg]{self.model}[/] not found\navailable{suffix}: {shown}", exit_code=1)

    def _openai_config(self, response_model: type[BaseModel]) -> dict[str, Any]:
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

    def _complete_json_openai(
        self,
        system: str,
        user: str,
        *,
        response_model: type[_M],
        retries: int,
        show_raw: bool,
    ) -> _M:
        conv = _Conversation(system, user)
        config = self._openai_config(response_model)

        for attempt in range(retries + 1):
            raw = (
                self._client.chat.completions.create(**config, messages=conv.messages).choices[0].message.content or ""
            )
            try:
                parsed = response_model.model_validate(json.loads(raw))
                if show_raw:
                    cout(f"LLM response:\n{raw}\n")
            except json.JSONDecodeError as e:
                if attempt < retries:
                    conv.llm(raw)
                    conv.usr("That was not valid JSON. Respond with ONLY valid JSON, no other text.")
                    continue
                msg = f"LLM returned invalid JSON after {retries + 1} attempts: {raw[:200]}"
                raise ValueError(msg) from e
            else:
                return parsed

        raise ValueError("unreachable")

    def _complete_json_gemini(
        self,
        system: str,
        user: str,
        *,
        response_model: type[_M],
        retries: int,
        show_raw: bool,
    ) -> _M:
        from google.genai import types as gtypes  # noqa: PLC0415

        config = gtypes.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=response_model.model_json_schema(),
        )
        contents = user

        for attempt in range(retries + 1):
            response = self._gemini_client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            raw = response.text or ""
            try:
                parsed = response_model.model_validate(json.loads(raw))
            except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001
                if attempt < retries:
                    contents = [
                        gtypes.Content(role="user", parts=[gtypes.Part(text=user)]),
                        gtypes.Content(role="model", parts=[gtypes.Part(text=raw)]),
                        gtypes.Content(
                            role="user",
                            parts=[gtypes.Part(text="That was not valid JSON. Respond with ONLY valid JSON.")],
                        ),
                    ]
                    continue
                msg = f"Gemini returned invalid JSON after {retries + 1} attempts: {raw[:200]}"
                raise ValueError(msg) from e
            else:
                if show_raw:
                    cout(f"LLM response:\n{raw}\n")
                return parsed

        raise ValueError("unreachable")

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        response_model: type[_M],
        retries: int = 2,
        show_raw: bool = False,
    ) -> _M:
        if self.service == "gemini":
            return self._complete_json_gemini(
                system,
                user,
                response_model=response_model,
                retries=retries,
                show_raw=show_raw,
            )
        return self._complete_json_openai(
            system,
            user,
            response_model=response_model,
            retries=retries,
            show_raw=show_raw,
        )
