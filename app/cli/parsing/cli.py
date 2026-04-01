import sys
from types import SimpleNamespace

from app.cli.parsing.display import phelp, pver
from app.cli.parsing.opt import Opt, SpecialOpt
from app.cli.parsing.parse import parse, parse_special_opts


def parse_with_opts(*opts: Opt) -> SimpleNamespace:
    special_opts = (
        SpecialOpt(
            short="-h",
            long="--help",
            desc="Show this help message & exit",
            triggers=lambda: phelp(*special_opts, *opts),
        ),
        SpecialOpt(short="-V", long="--version", desc="Show program's version number & exit", triggers=pver),
        *(opt for opt in opts if isinstance(opt, SpecialOpt)),
    )

    args = sys.argv[1:]
    parse_special_opts(special_opts, args)
    return parse(opts, args)
