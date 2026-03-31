#!/usr/bin/env python3

"""Load/Download initial dataset from huggingface."""

from __future__ import annotations

from functools import partial
from typing import Final

from common.utils.dataset import load_ds

DS_NAME: Final = "Suzhen/CodeChat-V2.0"
DS_REVISION: Final = "09dacf311596f8214075878600dcb60e5bcd7eb4"  # 2025-09-20

load_codechat_v2: Final = partial(load_ds, DS_NAME, revision=DS_REVISION)


if __name__ == "__main__":
    from common.utils.cache import graceful_exit

    with graceful_exit("Download cancelled"):
        load_codechat_v2()
