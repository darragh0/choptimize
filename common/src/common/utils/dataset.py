"""Dataset download utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import List, Value, Version, load_dataset
from huggingface_hub.errors import HfHubHTTPError

from common.utils.cache import CACHE_DIR
from common.utils.console import cout, cwarn
from common.utils.display import pretty_path

if TYPE_CHECKING:
    from datasets import Dataset


def fmt_type(feat: List | Value | dict) -> str:
    if isinstance(feat, Value):
        return feat.dtype
    if isinstance(feat, List):
        return rf"list\[{fmt_type(feat.feature)}]"
    return type(feat).__name__


def inner_fields(feat: List | Value | dict) -> dict | None:
    if isinstance(feat, List):
        return inner_fields(feat.feature)
    if isinstance(feat, dict):
        return feat
    return None


def show_oview(ds: Dataset, ds_name: str) -> None:
    location = pretty_path(CACHE_DIR / f"{ds_name}.parquet")

    cout(f"[dim]Dataset[/]    {ds_name!r}")
    cout(f"[dim]Location[/]   {location}")
    cout(f"[dim]Rows[/]       {len(ds):,}")
    cout("\n[dim]Features[/]")

    maxlen = lambda s: len(max(s, key=len))  # noqa: E731
    w = maxlen(ds.features)
    zpad = len(str(len(ds.features)))

    for i, (name, feat) in enumerate(ds.features.items()):
        typ = fmt_type(feat)
        cout(f"  {i:0{zpad}}  {name:<{w}}\t[cyan]{typ}[/]")

        if inner := inner_fields(feat):
            items = list(inner.items())
            iw = max({w, maxlen(inner)}) - 4
            for j, (sub_name, sub_feat) in enumerate(items):
                pre = "└─" if j == len(items) - 1 else "├─"
                styp = fmt_type(sub_feat)
                cout(f"    {' ' * zpad}[dim]  {pre}[/] {sub_name:<{iw}}\t  [cyan]{styp}[/]")
    cout()


def load_ds(
    name: str,
    *,
    revision: str | Version | None = None,
    split: str = "train",
    overview: bool = True,
    status: str | None = "[dim]Loading {name} dataset[/]",
) -> Dataset:
    try:
        if status is not None:
            with cout.status(status.format(name=name) if "{name}" in status else status, spinner="flip"):
                ds = load_dataset(name, revision=revision, cache_dir=str(CACHE_DIR))[split]
        else:
            ds = load_dataset(name, revision=revision, cache_dir=str(CACHE_DIR))[split]
    except HfHubHTTPError as e:
        cwarn(f"failed to load dataset from huggingface: {e}")
        raise RuntimeError("unreachable") from e

    if overview:
        show_oview(ds, ds_name=name)
    return ds
