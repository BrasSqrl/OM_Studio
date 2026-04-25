"""Small file-derived caches keyed by path, size, and modified time."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import pandas as pd


def read_csv_cached(path: Path) -> pd.DataFrame:
    key = _file_cache_key(path)
    return _read_csv_cached(*key).copy(deep=True)


def read_dataset_columns_cached(path: Path) -> list[str]:
    key = _file_cache_key(path)
    return list(_read_dataset_columns_cached(*key))


def sha256_file_cached(path: Path) -> str:
    key = _file_cache_key(path)
    return _sha256_file_cached(*key)


def _file_cache_key(path: Path) -> tuple[str, int, int]:
    resolved = path.resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)


@lru_cache(maxsize=64)
def _read_csv_cached(path_text: str, _size_bytes: int, _mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(path_text)


@lru_cache(maxsize=256)
def _read_dataset_columns_cached(path_text: str, _size_bytes: int, _mtime_ns: int) -> tuple[str, ...]:
    path = Path(path_text)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return tuple(pd.read_csv(path, nrows=0).columns.astype(str).tolist())
    if suffix in {".xlsx", ".xls"}:
        return tuple(pd.read_excel(path, nrows=0).columns.astype(str).tolist())
    return ()


@lru_cache(maxsize=128)
def _sha256_file_cached(path_text: str, _size_bytes: int, _mtime_ns: int) -> str:
    digest = sha256()
    with Path(path_text).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
