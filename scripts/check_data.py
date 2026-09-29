#!/usr/bin/env python3
"""Validate externally supplied math Parquet files without GPU/model imports."""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    for path in (args.train, args.evaluation):
        if not path.is_file():
            parser.error(f"Missing data file: {path}. See docs/data.md.")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        parser.error("pyarrow is required. Run bash run.sh setup or use the training environment.")
    valid = True
    for path, expected in ((args.train, 57046), (args.evaluation, 1590)):
        parquet = pq.ParquetFile(path)
        count = parquet.metadata.num_rows
        missing = {"prompt", "reward_model", "data_source"} - set(parquet.schema_arrow.names)
        ok = count == expected and not missing
        print(f"{path.name}: {count} rows (expected {expected}); {'OK' if ok else 'MISMATCH'}")
        if missing:
            print(f"Missing columns: {', '.join(sorted(missing))}", file=sys.stderr)
        valid = valid and ok
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
