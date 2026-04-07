from __future__ import annotations

from shutil import get_terminal_size
from typing import TYPE_CHECKING

from common.utils.console import cout, cwarn

from app.cli.display import display_result
from app.cli.parsing.cli import Cmd, Opt, parse_with_opts

if TYPE_CHECKING:
    from types import SimpleNamespace


def _launch_web(_: SimpleNamespace) -> None:
    import uvicorn  # noqa: PLC0415

    from app.web.app import app as web_app  # noqa: PLC0415

    uvicorn.run(web_app, host="127.0.0.1", port=8000)


def cli() -> None:
    cfg = parse_with_opts(
        Opt(short="-i", long="--improve", desc="Generate an improved version of the prompt"),
        Opt(short="-s", long="--service", desc="LLM service", takes=("service", ("ollama", "openai", "gemini"))),
        Opt(short="-m", long="--model", desc="LLM model name", takes=("model", str)),
        Opt(short="-k", long="--api-key", desc="LLM API key", takes=("key", str)),
        cmds=(Cmd("web", "Launch web server", run=_launch_web),),
    )

    half_cols = get_terminal_size().columns / 2
    prompt_clamped = f"{cfg.prompt}..." if len(cfg.prompt) > half_cols else cfg.prompt

    try:
        with cout.status(f'Analyzing ([dim]"{prompt_clamped}"[/])', spinner="line"):
            from app.engine import Engine  # noqa: PLC0415

            result = Engine(
                service=cfg.service,
                model=cfg.model,
                api_key=cfg.api_key,
            ).analyze(
                cfg.prompt,
                improve=cfg.improve,
                show_raw=False,
            )

        display_result(result)

    except ValueError as e:
        cwarn((e), exit_code=1)
        return

    except KeyboardInterrupt:
        cwarn("analysis interrupted", exit_code=130)


if __name__ == "__main__":
    cli()
