import json
import os
from datetime import datetime
from pathlib import Path

HISTORY_DIR = Path("bias_history")


def save_scan_result(
    dataset_name: str,
    bias_results: list,
    profile: dict,
    label: str = None,
) -> str:
    """Save a scan result to disk with a timestamp."""
    HISTORY_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = label or timestamp
    safe_name = dataset_name.replace(" ", "_").replace("/", "_")

    record = {
        "dataset_name": dataset_name,
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "profile_shape": profile.get("shape", {}),
        "bias_summary": [
            {
                "bias_type": r["bias_type"],
                "severity": r["severity"],
                "detected": r["detected"],
                "affected_column_count": len(r["affected_columns"]),
            }
            for r in bias_results
        ],
        "bias_results": bias_results,
    }

    filepath = HISTORY_DIR / f"{safe_name}_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(record, f, indent=2, default=str)

    return str(filepath)


def load_history(dataset_name: str = None) -> list:
    """Load all saved scan results, optionally filtered by dataset name."""
    if not HISTORY_DIR.exists():
        return []

    records = []
    for filepath in sorted(HISTORY_DIR.glob("*.json")):
        try:
            with open(filepath) as f:
                record = json.load(f)
            if dataset_name is None or record.get("dataset_name") == dataset_name:
                records.append(record)
        except Exception:
            continue

    return sorted(records, key=lambda r: r["timestamp"])


def list_tracked_datasets() -> list:
    """Return list of unique dataset names that have history."""
    if not HISTORY_DIR.exists():
        return []
    names = set()
    for filepath in HISTORY_DIR.glob("*.json"):
        try:
            with open(filepath) as f:
                record = json.load(f)
            names.add(record.get("dataset_name", "unknown"))
        except Exception:
            continue
    return sorted(names)


def build_trend_dataframe(history: list) -> "pd.DataFrame":
    """Turn history records into a flat DataFrame for plotting."""
    import pandas as pd
    rows = []
    for record in history:
        base = {
            "label": record["label"],
            "timestamp": record["timestamp"],
            "dataset_name": record["dataset_name"],
            "rows": record.get("profile_shape", {}).get("rows", None),
        }
        for b in record.get("bias_summary", []):
            row = {**base}
            row["bias_type"] = b["bias_type"]
            row["severity"] = b["severity"]
            row["detected"] = b["detected"]
            row["affected_column_count"] = b["affected_column_count"]
            row["severity_score"] = {"high": 3, "medium": 2, "low": 1, "unknown": 0}.get(
                b["severity"], 0
            )
            rows.append(row)
    return pd.DataFrame(rows)