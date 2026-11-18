"""Paths, all overridable by environment variable.

Defaults resolve relative to the repository checkout, so a fresh clone runs
with no configuration at all. Set the variables only to keep data or the
ledger somewhere else.

    HALLMARK_DATA_DIR   root for frozen datasets, probe set, and cache
                        (default: <repo>/data)
    HALLMARK_DB         SQLite ledger path (default: $HALLMARK_DATA_DIR/hallmark.db)
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return Path(os.environ.get("HALLMARK_DATA_DIR") or REPO_ROOT / "data")


def frozen_dir() -> Path:
    return data_dir() / "frozen"


def probe_dir() -> Path:
    return data_dir() / "probe_set"


def cache_dir() -> Path:
    return data_dir() / "cache"


def db_path() -> Path:
    return Path(os.environ.get("HALLMARK_DB") or data_dir() / "hallmark.db")
