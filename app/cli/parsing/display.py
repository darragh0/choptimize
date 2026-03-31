from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final

from common.utils.console import CustomConsole, CustomErrConsole
from rich.theme import Theme

from app import __desc__, __prog__, __version__

if TYPE_CHECKING:
    from app.cli.parsing.opt import Opt

N_INDENT: Final = 2
_LJUST_GAP: Final = 2


_APP_THEME: Final = Theme(
    {
        "prog": "bold cyan",
        "ver": "green",
        "arg": "cyan",
        "desc": "bright_white",
        "metavar": "dim cyan",
        "help_title": "bold green",
        "tip": "green",
    },
)

_VERSION: Final = f"[prog]{__prog__}[/] [ver]v{__version__}[/]"

_HELP: Final = f"""\
{__desc__}

[help_title]Args[/]:
{" " * N_INDENT}[arg]prompt[/]   Prompt to optimize

[help_title]Options[/]:
{{opt_content}}\
"""

_USAGE: Final = f"\n[help_title]Usage:[/] [prog]{__prog__}[/] [metavar]<options>[/] [arg]prompt...[/]"

_COMMON_ERR: str = (
    f"\n{' ' * N_INDENT}[tip]tip:[/] use [arg]--[/] before your prompt to "
    "include flags literally, e.g. [arg]-- my prompt --like-this[/]"
)


def _ljust(opts: tuple[Opt, ...]) -> int:
    return (
        N_INDENT
        + _LJUST_GAP
        + max((len(o.short) + 2 if o.short else 0) + len(o.long) + (1 + len(o.takes) if o.takes else 0) for o in opts)
    )


def phelp(*opts: Opt) -> None:
    lj = _ljust(opts)
    cout(_HELP.format(opt_content="\n".join(o.help(lj) for o in opts)))
    sys.exit(0)


def pusage(*, hint: bool = False) -> None:
    if hint:
        cout(_COMMON_ERR)
    cout(_USAGE)
    sys.exit(2)


def pver() -> None:
    cout(_VERSION)
    sys.exit(0)


cout = CustomConsole(theme=_APP_THEME)
cerr = CustomErrConsole(theme=_APP_THEME)
