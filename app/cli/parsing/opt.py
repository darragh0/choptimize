from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, LiteralString

from annotated_types import Ge

from app.cli.parsing.display import N_INDENT

if TYPE_CHECKING:
    from collections.abc import Callable

type StringChoices = tuple[LiteralString, ...]
type AllowedOptType = type[str | int | float] | StringChoices


@dataclass
class _OptValue:
    metavar: str
    type: AllowedOptType | StringChoices

    def __str__(self) -> str:
        return f"[metavar]<{self.metavar}>[/]"

    def __len__(self) -> int:
        return len(self.metavar) + 3


class Opt:
    short: str | None
    long: str
    desc: str
    takes: _OptValue | None

    def __init__(
        self,
        long: str,
        desc: str,
        short: str | None = None,
        takes: tuple[str, AllowedOptType] | type | None = None,
    ) -> None:
        self.short = short
        self.long = long
        self.desc = desc

        if isinstance(takes, type):
            self.takes = _OptValue(takes.__name__, takes)
        elif isinstance(takes, tuple):
            self.takes = _OptValue(*takes)
        else:
            self.takes = None

    def help(self, ljust: Annotated[int, Ge(0)]) -> str:
        short_len = len(self.short) + 2 if self.short else 4  # 4 = width of "-x, "
        n_space = ljust - N_INDENT - short_len - len(self.long) - (len(self.takes) + 1 if self.takes is not None else 1)

        return (
            f"{' ' * (N_INDENT + (0 if self.short else 4))}"
            f"{f'[arg]{self.short}[/], ' if self.short else ''}"
            f"[arg]{self.long}[/]"
            f"{f' {self.takes}' if self.takes is not None else ''}"
            f"{' ' * n_space}"
            f"[desc]{self.desc}[/]"
        )


class SpecialOpt(Opt):
    triggers: Callable[[], None]

    def __init__(
        self,
        long: str,
        desc: str,
        short: str | None = None,
        takes: tuple[str, AllowedOptType] | type | None = None,
        triggers: Callable[[], None] = lambda: None,
    ) -> None:
        super().__init__(long, desc, short, takes)
        self.triggers = triggers
