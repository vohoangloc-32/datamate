"""Parallel pre-caching for all businesses in train and public_test."""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical import load_canonical_business


def cache_single_biz(item):
    split, biz_id = item
    try:
        load_canonical_business(split, biz_id, cache_dir="artifacts/cache", use_cache=True)
        return (split, biz_id, "OK")
    except Exception as exc:
        return (split, biz_id, f"ERR: {exc}")


def main():
    items = []
    for split in ["public_test", "train"]:
        p = Path(split) / "data"
        if not p.exists():
            continue
        for b in sorted(os.listdir(p)):
            if (p / b).is_dir():
                items.append((split, b))

    print(f"Total businesses to pre-cache: {len(items)}")
    workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for split, biz_id, status in executor.map(cache_single_biz, items):
            print(f"[{status}] {split}/{biz_id}")


if __name__ == "__main__":
    main()
