"""Display utils & console handles (shorthands for `cout.print` & `cerr.print`)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

from rich.console import Console

if TYPE_CHECKING:
    from pandas import DataFrame
    from rich.console import JustifyMethod, OverflowMethod
    from rich.style import Style


class _RichConsolePrintKwargs(TypedDict, total=False):
    """Typed keyword arguments for `rich.console.print`."""

    sep: str
    end: str
    style: str | Style | None
    justify: JustifyMethod | None
    overflow: OverflowMethod | None
    no_wrap: bool | None
    emoji: bool | None
    markup: bool | None
    highlight: bool | None
    width: int | None
    height: int | None
    crop: bool
    soft_wrap: bool | None
    new_line_start: bool


class _Console(Console):
    def __call__(
        self,
        *objects: Any,  # noqa: ANN401
        **kwargs: Unpack[_RichConsolePrintKwargs],
    ) -> None:
        self.print(*objects, **kwargs)


class _ErrConsole(Console):
    def __call__(
        self,
        *objects: Any,  # noqa: ANN401
        exit_code: int | None = None,
        prefix: str = "[bold red]error:[/]",
        **kwargs: Unpack[_RichConsolePrintKwargs],
    ) -> None:
        self.print(prefix, *objects, **kwargs)

        if exit_code is not None:
            sys.exit(exit_code)


def pretty_path(path: Path, /) -> str:
    return f"[magenta]{f'{path.resolve()}'.replace(str(Path.home()), '~', 1)}[/]"


def show_df_overview(df: DataFrame, /) -> None:
    """Show DataFrame overview."""
    cout("DataFrame Info")
    cout(f"  [dim]Shape[/]   {df.shape}")
    cout(f"  [dim]Models[/]  {df['model'].nunique()} unique")
    cout("  [dim]Columns[/]")

    col_width = max(len(c) for c in df.columns)
    zpad = len(str(df.shape[1]))

    for i, col in enumerate(df.columns):
        cout(f"    {i:0{zpad}}  {col:<{col_width}}\t[cyan]{df[col].dtype}[/]")


cout = _Console()
cerr = _ErrConsole(stderr=True)
