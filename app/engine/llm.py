import json
import os

from common.utils.console import cout
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

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

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        retries: int = 2,
        required_keys: tuple[str, ...] = (),
        show_raw: bool,
    ) -> dict:
        messages: list[ChatCompletionMessageParam] = [
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
            if show_raw:
                cout("Raw LLM response:")
                cout(f"```\n{raw}\n```")

            parsed: dict | None = None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                if attempt < retries:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": "That was not valid JSON. Respond with ONLY valid JSON, no other text.",
                        }
                    )
                    continue
                msg = f"LLM returned invalid JSON after {retries + 1} attempts: {raw[:200]}"
                raise ValueError(msg) from e

            if required_keys and parsed is not None:
                missing = [k for k in required_keys if k not in parsed]
                if missing:
                    if attempt < retries:
                        messages.append({"role": "assistant", "content": raw})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Missing required keys: {', '.join(missing)}. "
                                    "Respond with ONLY valid JSON containing all required keys."
                                ),
                            }
                        )
                        continue
                    msg = f"LLM response missing keys {missing} after {retries + 1} attempts"
                    raise ValueError(msg)

            return parsed  # type: ignore[return-value]
        return None  # type: ignore[return-value]
