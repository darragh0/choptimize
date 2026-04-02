from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final

from common.utils.console import CustomConsole, CustomErrConsole
from rich.theme import Theme

from app import __desc__, __prog__, __version__

if TYPE_CHECKING:
    from collections.abc import Sequence

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

_USAGE: Final = f"\n[help_title]Usage:[/] [prog]{__prog__}[/] [metavar]<options>[/] [arg]prompt...[/]"

_COMMON_ERR: str = (
    f"\n{' ' * N_INDENT}[tip]tip:[/] use [arg]--[/] before your prompt to "
    "include flags literally, e.g. [arg]-- my prompt --like-this[/]"
)


_I: Final = " " * N_INDENT


def _ljust(opts: tuple[Opt, ...]) -> int:
    return (
        N_INDENT
        + _LJUST_GAP
        + max((len(o.short) + 2 if o.short else 0) + len(o.long) + (1 + len(o.takes) if o.takes else 0) for o in opts)
    )


def phelp(*opts: Opt, cmds: Sequence[tuple[str, str]] = ()) -> None:
    lj = _ljust(opts)
    lines: list[str] = [__desc__, ""]

    lines.append("[help_title]Usage[/]:")
    lines.append(f"{_I}[prog]{__prog__}[/] [metavar]\\[options][/] [arg]prompt...[/]")
    if cmds:
        lines.append(f"{_I}[prog]{__prog__}[/] [metavar]<command>[/] [metavar]\\[options][/]")
    lines.append("")

    lines.append("[help_title]Args[/]:")
    lines.append(f"{_I}[arg]prompt[/]   Prompt to optimize")
    lines.append("")

    if cmds:
        cmd_pad = max(len(n) for n, _ in cmds) + _LJUST_GAP
        lines.append("[help_title]Commands[/]:")
        for name, desc in cmds:
            lines.append(f"{_I}[arg]{name}[/]{' ' * (cmd_pad - len(name))}[desc]{desc}[/]")
        lines.append("")

    lines.append("[help_title]Options[/]:")
    lines.extend(o.help(lj) for o in opts)

    cout("\n".join(lines), end="\n\n")
    sys.exit(0)


def pcmd_help(name: str, desc: str, *opts: Opt) -> None:
    lines: list[str] = [desc, ""]

    lines.append("[help_title]Usage[/]:")
    lines.append(f"{_I}[prog]{__prog__}[/] [arg]{name}[/] [metavar]\\[options][/]")
    lines.append("")

    if opts:
        lj = _ljust(opts)
        lines.append("[help_title]Options[/]:")
        lines.extend(o.help(lj) for o in opts)

    cout("\n".join(lines), end="\n\n")
    sys.exit(0)


def pusage(*, hint: bool = False) -> None:
    if hint:
        cout(_COMMON_ERR)
    cout(_USAGE)
    sys.exit(2)


def pver() -> None:
    cout(_VERSION)
    sys.exit(0)


cout = CustomConsole(theme=_APP_THEME, highlight=False)
cerr = CustomErrConsole(theme=_APP_THEME)
