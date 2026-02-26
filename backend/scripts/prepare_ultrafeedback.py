"""
Download UltraFeedback and export to JSONL for ingestion.

Each instruction has ~4 model completions with GPT-4 annotations.
This script flattens them into one JSONL row per completion, adding
scenario_id (groups candidates) and candidate_source (model name).

Usage:
  pip install datasets
  python scripts/prepare_ultrafeedback.py [--dev-size 2000] [--test-size 500] [--ab-size 300]
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare UltraFeedback data for QA Studio")
    parser.add_argument("--dev-size", type=int, default=2000, help="Dev split size (scenarios)")
    parser.add_argument("--test-size", type=int, default=500, help="Test split size (scenarios)")
    parser.add_argument("--ab-size", type=int, default=300, help="AB eval split size (scenarios)")
    parser.add_argument("--output-dir", default="../sample_data", help="Output directory")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed. Run: pip install datasets>=2.14.0",
              file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_needed = args.dev_size + args.test_size + args.ab_size

    print(f"Loading UltraFeedback dataset (streaming, need {total_needed} scenarios)...")
    ds = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)

    scenarios = []
    for i, example in enumerate(ds):
        if len(scenarios) >= total_needed:
            break
        scenarios.append(example)
        if (i + 1) % 500 == 0:
            print(f"  Loaded {len(scenarios)}/{total_needed} scenarios...")

    print(f"Loaded {len(scenarios)} scenarios total")

    # Split into dev / test / ab_eval
    dev_scenarios = scenarios[:args.dev_size]
    test_scenarios = scenarios[args.dev_size:args.dev_size + args.test_size]
    ab_scenarios = scenarios[args.dev_size + args.test_size:]

    splits = [
        ("ultrafeedback_dev.jsonl", dev_scenarios),
        ("ultrafeedback_test.jsonl", test_scenarios),
        ("ultrafeedback_ab_eval.jsonl", ab_scenarios),
    ]

    # Build index lookup for O(1) scenario numbering
    scenario_index = {id(s): i for i, s in enumerate(scenarios)}

    for filename, split_scenarios in splits:
        path = output_dir / filename
        row_count = 0
        with open(path, "w") as f:
            for scenario in split_scenarios:
                idx = scenario_index[id(scenario)]
                source = scenario.get("source", "unknown")
                instruction = scenario.get("instruction", "")
                completions = scenario.get("completions", [])
                scenario_id = f"uf_{source}_{idx}"

                for comp in completions:
                    model = comp.get("model", "unknown")
                    response_text = comp.get("response", "")
                    annotations = comp.get("annotations", {})
                    custom_system_prompt = comp.get("custom_system_prompt", "")

                    # Extract annotation scores
                    annotation_scores = {}
                    overall_score = None
                    critique = ""
                    for dim_name, dim_data in annotations.items():
                        if isinstance(dim_data, dict):
                            rating = dim_data.get("Rating", dim_data.get("rating"))
                            if rating is not None:
                                try:
                                    annotation_scores[dim_name] = int(rating)
                                except (ValueError, TypeError):
                                    annotation_scores[dim_name] = rating
                            rationale = dim_data.get("Rationale", dim_data.get("rationale", ""))
                            if rationale:
                                critique += f"{dim_name}: {rationale}\n"

                    if annotation_scores:
                        numeric_scores = [v for v in annotation_scores.values() if isinstance(v, (int, float))]
                        if numeric_scores:
                            overall_score = round(sum(numeric_scores) / len(numeric_scores), 1)

                    # Sanitize model name for ID
                    model_safe = model.replace("/", "-").replace(" ", "_")

                    record = {
                        "id": f"uf_{source}_{idx}_{model_safe}",
                        "scenario_id": scenario_id,
                        "candidate_source": model,
                        "system_prompt": custom_system_prompt or None,
                        "question": instruction,
                        "response": response_text,
                        "metadata": {
                            "dataset": "ultrafeedback",
                            "source": source,
                            "overall_score": overall_score,
                            "annotations": annotation_scores,
                            "critique": critique.strip(),
                        },
                    }
                    f.write(json.dumps(record) + "\n")
                    row_count += 1

        print(f"Wrote {row_count} rows ({len(split_scenarios)} scenarios) -> {path}")

    print(f"\nDone! Files in {output_dir}/")
    print(f"  ultrafeedback_dev.jsonl     ({len(dev_scenarios)} scenarios)")
    print(f"  ultrafeedback_test.jsonl    ({len(test_scenarios)} scenarios)")
    print(f"  ultrafeedback_ab_eval.jsonl ({len(ab_scenarios)} scenarios)")
    print(f"\nEach scenario has ~4 candidate rows (one per model completion).")


if __name__ == "__main__":
    main()
