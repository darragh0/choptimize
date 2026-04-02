from __future__ import annotations

import sys
from typing import TYPE_CHECKING, NamedTuple

from app.cli.parsing.display import pcmd_help, phelp, pver
from app.cli.parsing.opt import Opt, SpecialOpt
from app.cli.parsing.parse import parse, parse_special_opts

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import SimpleNamespace


class Cmd(NamedTuple):
    name: str
    desc: str
    run: Callable[[SimpleNamespace], None]
    opts: tuple[Opt, ...] = ()


def parse_with_opts(*opts: Opt, cmds: tuple[Cmd, ...] = ()) -> SimpleNamespace:
    cmd_info = [(c.name, c.desc) for c in cmds]

    special_opts = (
        SpecialOpt(
            short="-h",
            long="--help",
            desc="Show this help message & exit",
            triggers=lambda: phelp(*special_opts, *opts, cmds=cmd_info),
        ),
        SpecialOpt(short="-V", long="--version", desc="Show program's version number & exit", triggers=pver),
    )

    args = sys.argv[1:]

    matched = next((c for c in cmds if c.name == args[0]), None) if args and not args[0].startswith("-") else None
    if matched is not None:
        cmd_args = args[1:]
        cmd_special: tuple[SpecialOpt, ...] = (
            SpecialOpt(
                short="-h",
                long="--help",
                desc="Show this help message & exit",
                triggers=lambda: pcmd_help(matched.name, matched.desc, *cmd_special, *matched.opts),
            ),
        )
        parse_special_opts(cmd_special, cmd_args)
        cfg = parse(matched.opts, cmd_args, require_prompt=False)
        matched.run(cfg)
        sys.exit(0)

    parse_special_opts(special_opts, args)
    return parse(opts, args)
