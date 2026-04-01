from shutil import get_terminal_size

import uvicorn
from common.utils.console import cout, cwarn

from app.cli.display import display_result
from app.cli.parsing.cli import Opt, parse_with_opts
from app.cli.parsing.opt import SpecialOpt


def _launch_web() -> None:
    from app.web.app import app as web_app  # noqa: PLC0415

    uvicorn.run(web_app, host="127.0.0.1", port=8000)


def cli() -> None:
    cfg = parse_with_opts(
        Opt(short="-v", long="--verbose", desc="Enable verbose output"),
        Opt(short="-i", long="--improve", desc="Generate an improved version of the prompt"),
        Opt(short="-m", long="--model", desc="LLM model name", takes=("model", str)),
        Opt(long="--llm-url", desc="LLM API base URL", takes=("url", str)),
        SpecialOpt(short="-w", long="--web", desc="Launch web server (ignores prompt arg)", triggers=_launch_web),
    )

    half_cols = get_terminal_size().columns / 2
    prompt_clamped = f"{cfg.prompt}..." if len(cfg.prompt) > half_cols else cfg.prompt

    try:
        with cout.status(f'Analyzing\n  [dim]>> [green]"{prompt_clamped}"[/][/]', spinner="line"):
            from app.engine import Engine  # noqa: PLC0415

            engine = Engine(model=cfg.model, llm_url=cfg.llm_url)
            result = engine.analyze(cfg.prompt, improve=cfg.improve)

        display_result(result, verbose=cfg.verbose)
    except KeyboardInterrupt:
        cwarn("analysis interrupted", exit_code=130)


if __name__ == "__main__":
    cli()
