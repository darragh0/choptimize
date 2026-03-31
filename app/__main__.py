import time
from shutil import get_terminal_size

from common.utils.console import cout, cwarn

from app.cli.parsing.cli import Opt, parse_with_opts


def cli() -> None:
    cfg = parse_with_opts(
        Opt(short="-v", long="--verbose", desc="Enable verbose output"),
    )

    half_cols = get_terminal_size().columns / 2
    prompt_clamped = f"{cfg.prompt}..." if len(cfg.prompt) > half_cols else cfg.prompt

    try:
        with cout.status(f'Optimizing\n  [dim]>> [green]"{prompt_clamped}"[/][/]', spinner="line"):
            time.sleep(5)
    except KeyboardInterrupt:
        cwarn("optimization interrupted", exit_code=130)


if __name__ == "__main__":
    cli()
