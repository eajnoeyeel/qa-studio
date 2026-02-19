"""
Transform IT Support Tickets 100k CSV → JSONL for CS QA Studio ingestion.

Maps:
  - initial_message → conversation[0] (user message)
  - agent_first_reply → candidate_response
  - issue_type + product_area → taxonomy label
  - Extra metadata preserved for analysis

Usage:
  python scripts/prepare_data.py [--limit N] [--output PATH]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# ============== Taxonomy Mapping ==============

# Map (issue_type, product_area) → taxonomy label
# Uses issue_type as primary, product_area as disambiguator

ISSUE_TYPE_MAP = {
    "billing_problem": "billing_seats",       # billing issues → billing_seats (default)
    "account_access": "workspace_access",     # access issues → workspace_access
    "bug": "bug_report",                      # bugs → bug_report
    "feature_request": "feature_request",     # 1:1 mapping
    "security_concern": "permission_sharing", # security → permission_sharing
    "performance": "bug_report",              # performance issues → bug_report
    "how_to": "feature_request",              # how-to questions → feature_request
    "other": "bug_report",                    # catch-all
}

# Product area can override or refine the mapping
PRODUCT_AREA_OVERRIDES = {
    ("billing_problem", "billing"): "billing_seats",
    ("billing_problem", "data_export"): "billing_refund",
    ("account_access", "login_auth"): "login_sso",
    ("account_access", "api_integration"): "workspace_access",
    ("security_concern", "login_auth"): "login_sso",
    ("security_concern", "api_integration"): "permission_sharing",
    ("how_to", "data_export"): "import_export_sync",
    ("how_to", "api_integration"): "import_export_sync",
    ("how_to", "billing"): "billing_seats",
    ("bug", "login_auth"): "login_sso",
    ("bug", "billing"): "billing_seats",
    ("bug", "data_export"): "import_export_sync",
    ("performance", "login_auth"): "login_sso",
    ("performance", "api_integration"): "import_export_sync",
    ("other", "billing"): "billing_refund",
    ("other", "login_auth"): "login_sso",
    ("other", "data_export"): "import_export_sync",
}


def get_taxonomy_label(issue_type: str, product_area: str) -> str:
    """Map issue_type + product_area to taxonomy label."""
    key = (issue_type, product_area)
    if key in PRODUCT_AREA_OVERRIDES:
        return PRODUCT_AREA_OVERRIDES[key]
    return ISSUE_TYPE_MAP.get(issue_type, "bug_report")


def transform_row(row: dict) -> dict:
    """Transform a single CSV row to JSONL record for ingestion."""
    taxonomy_label = get_taxonomy_label(row["issue_type"], row["product_area"])

    return {
        "external_id": row["ticket_id"],
        "conversation": [
            {
                "role": "user",
                "content": row["initial_message"],
            }
        ],
        "candidate_response": row["agent_first_reply"],
        "metadata": {
            "source": "kaggle_it_support_100k",
            "taxonomy_label": taxonomy_label,
            "issue_type": row["issue_type"],
            "product_area": row["product_area"],
            "priority": row["priority"],
            "status": row["status"],
            "channel": row["channel"],
            "customer_segment": row["customer_segment"],
            "customer_sentiment": row["customer_sentiment"],
            "csat_score": int(row["csat_score"]) if row["csat_score"] else None,
            "sla_plan": row["sla_plan"],
            "resolution_summary": row["resolution_summary"],
            "resolution_time_hours": float(row["resolution_time_hours"]) if row["resolution_time_hours"] else None,
            "platform": row["platform"],
            "region": row["region"],
            "created_at": row["created_at"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Transform IT Support Tickets CSV to JSONL")
    parser.add_argument("--input", default="data/synthetic_it_support_tickets.csv",
                        help="Input CSV path (default: data/synthetic_it_support_tickets.csv)")
    parser.add_argument("--output", default="data/tickets.jsonl",
                        help="Output JSONL path (default: data/tickets.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of rows (default: all)")
    parser.add_argument("--stats", action="store_true",
                        help="Print taxonomy distribution stats")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track stats
    taxonomy_counts = {}
    total = 0
    skipped = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        reader = csv.DictReader(fin)
        for i, row in enumerate(reader):
            if args.limit and i >= args.limit:
                break

            # Skip rows with empty messages
            if not row.get("initial_message") or not row.get("agent_first_reply"):
                skipped += 1
                continue

            record = transform_row(row)
            fout.write(json.dumps(record) + "\n")

            label = record["metadata"]["taxonomy_label"]
            taxonomy_counts[label] = taxonomy_counts.get(label, 0) + 1
            total += 1

    print(f"✅ Transformed {total} tickets → {output_path}")
    if skipped:
        print(f"⚠️  Skipped {skipped} rows (empty messages)")

    if args.stats or True:  # Always show stats
        print(f"\n📊 Taxonomy Distribution:")
        for label, count in sorted(taxonomy_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {label:<25} {count:>6} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
