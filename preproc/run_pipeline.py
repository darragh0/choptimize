#!/usr/bin/env python3

"""Run full preprocessing pipeline: download -> filter -> static (syntax, then semantics)."""

from __future__ import annotations

from typing import Final

from utils.cache import graceful_exit
from utils.console import cout

STAGES: Final = (
    ("download", "Downloading dataset"),
    ("filter", "Filtering dataset"),
    ("syntax", "Syntactic analysis"),
    ("semantics", "Semantic analysis"),
)


def main() -> None:
    cout("[bold]Running preprocessing pipeline[/]")
    cout()

    for i, (module_name, label) in enumerate(STAGES, 1):
        cout(f"[bold cyan]Stage {i}/{len(STAGES)}:[/] {label}")
        module = __import__(module_name)
        module.main()
        cout()

    cout("[bold green]Pipeline complete[/]")


if __name__ == "__main__":
    with graceful_exit("pipeline stopped"):
        main()
