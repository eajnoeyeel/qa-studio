"""
Download a subset of OpenOrca and export to JSONL for ingestion.

Usage:
  pip install datasets
  python scripts/prepare_openorca.py [--dev-size 5000] [--test-size 2000] [--ab-size 1000]
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenOrca data for QA Studio")
    parser.add_argument("--dev-size", type=int, default=5000, help="Dev split size")
    parser.add_argument("--test-size", type=int, default=2000, help="Test split size")
    parser.add_argument("--ab-size", type=int, default=1000, help="AB eval split size")
    parser.add_argument("--output-dir", default="../sample_data", help="Output directory")
    parser.add_argument("--archive-old", action="store_true", default=True,
                        help="Archive old sample data files")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed. Run: pip install datasets>=2.14.0",
              file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Archive old data if present
    if args.archive_old:
        archive_dir = output_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        for old_file in output_dir.glob("tickets_*.jsonl"):
            dest = archive_dir / old_file.name
            if not dest.exists():
                old_file.rename(dest)
                print(f"Archived: {old_file.name} -> archive/")

    total_needed = args.dev_size + args.test_size + args.ab_size

    print(f"Loading OpenOrca dataset (streaming, need {total_needed} rows)...")
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)

    rows = []
    for i, example in enumerate(ds):
        if len(rows) >= total_needed:
            break
        rows.append(example)
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {len(rows)}/{total_needed}...")

    print(f"Loaded {len(rows)} rows total")

    # Split into dev / test / ab_eval
    dev_rows = rows[:args.dev_size]
    test_rows = rows[args.dev_size:args.dev_size + args.test_size]
    ab_rows = rows[args.dev_size + args.test_size:]

    splits = [
        ("openorca_dev.jsonl", dev_rows),
        ("openorca_test.jsonl", test_rows),
        ("openorca_ab_eval.jsonl", ab_rows),
    ]

    for filename, split_rows in splits:
        path = output_dir / filename
        with open(path, "w") as f:
            for row in split_rows:
                record = {
                    "id": row.get("id", ""),
                    "system_prompt": row.get("system_prompt", ""),
                    "question": row.get("question", ""),
                    "response": row.get("response", ""),
                }
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(split_rows)} rows -> {path}")

    print(f"\nDone! Files in {output_dir}/")
    print(f"  openorca_dev.jsonl     ({len(dev_rows)} rows)")
    print(f"  openorca_test.jsonl    ({len(test_rows)} rows)")
    print(f"  openorca_ab_eval.jsonl ({len(ab_rows)} rows)")


if __name__ == "__main__":
    main()
