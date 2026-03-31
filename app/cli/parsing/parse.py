from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

from app.cli.parsing.display import cerr, pusage

if TYPE_CHECKING:
    from app.cli.parsing.opt import Opt, SpecialOpt


def _init_parsed(opts: tuple[Opt, ...]) -> SimpleNamespace:
    return SimpleNamespace(**{opt.long.lstrip("-").replace("-", "_"): None if opt.takes else False for opt in opts})


def _init_opt_map(opts: tuple[Opt, ...]) -> dict[str, Opt]:
    return {flag: opt for opt in opts for flag in (opt.short, opt.long) if flag is not None}


def parse_special_opts(special_opts: tuple[SpecialOpt, ...], args: list[str] = sys.argv[1:]) -> None:
    for spopt in special_opts:
        if spopt.long in args or (spopt.short is not None and spopt.short in args):
            spopt.triggers()


def _expand_args(args: list[str], opt_map: dict[str, Opt]) -> list[str]:
    expanded = []
    for i, arg in enumerate(args):
        if arg == "--":
            expanded.extend(args[i:])
            break
        if len(arg) > 2 and arg[0] == "-" and arg[1] != "-":
            has_positionals = any(not a.startswith("-") for a in args[:i])
            for j, ch in enumerate(arg[1:]):
                flag = f"-{ch}"
                if flag not in opt_map:
                    cerr(f"unknown option [arg]{flag}[/] (in [arg]{arg}[/])")
                    pusage(hint=has_positionals)
                if opt_map[flag].takes is not None and j < len(arg) - 2:
                    cerr(f"[arg]{flag}[/] takes a value; must be last in a combined option")
                    pusage(hint=has_positionals)
                expanded.append(flag)
        else:
            expanded.append(arg)
    return expanded


def parse(opts: tuple[Opt, ...], args: list[str]) -> SimpleNamespace:
    parsed, opt_map = _init_parsed(opts), _init_opt_map(opts)
    positionals: list[str] = []
    it = iter(_expand_args(args, opt_map))

    for arg in it:
        if arg == "--":
            positionals.extend(it)
            break

        if opt := opt_map.get(arg):
            name = opt.long.lstrip("-").replace("-", "_")
            if opt.takes is not None:
                if (raw := next(it, None)) is None:
                    cerr(f"[arg]{arg}[/] requires [metavar]<{opt.takes.metavar}>[/]")
                    pusage()
                    raise RuntimeError("unreachable")
                try:
                    setattr(parsed, name, opt.takes.type(raw))
                except (ValueError, TypeError):
                    cerr(f"[arg]{arg}[/] expected [green]{opt.takes.type.__name__}[/], got {raw!r}")
                    pusage()
            else:
                setattr(parsed, name, True)
        elif arg.startswith("-"):
            cerr(f"unknown option [arg]{arg}[/]")
            pusage(hint=bool(positionals))
        else:
            positionals.append(arg)
    if not positionals:
        cerr("no prompt given")
        pusage()
        raise RuntimeError("unreachable")
    parsed.prompt = " ".join(positionals)
    return parsed
