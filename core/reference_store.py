import json
from pathlib import Path

REFERENCE_DIR = Path("reference_distributions")

# Built-in defaults
BUILTIN_REFERENCES = {
    "US Census — Race": {
        "col_hint": "race",
        "distribution": {
            "white": 0.595,
            "black": 0.134,
            "hispanic": 0.186,
            "asian": 0.059,
            "other": 0.026,
        }
    },
    "US Census — Gender": {
        "col_hint": "gender",
        "distribution": {
            "male": 0.495,
            "female": 0.495,
            "other": 0.01,
        }
    },
    "US Census — Age Groups": {
        "col_hint": "age_cat",
        "distribution": {
            "under 18": 0.22,
            "18-34": 0.22,
            "35-54": 0.26,
            "55+": 0.30,
        }
    },
}


def save_custom_reference(name: str, col_hint: str, distribution: dict) -> str:
    """Save a custom reference distribution to disk."""
    REFERENCE_DIR.mkdir(exist_ok=True)
    total = sum(distribution.values())
    normalized = {k: round(v / total, 6) for k, v in distribution.items()}
    record = {
        "name": name,
        "col_hint": col_hint.lower().strip(),
        "distribution": normalized,
        "custom": True,
    }
    filepath = REFERENCE_DIR / f"{name.replace(' ', '_')}.json"
    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)
    return str(filepath)


def load_all_references() -> dict:
    """Return all references: builtins + any saved custom ones."""
    refs = dict(BUILTIN_REFERENCES)
    if not REFERENCE_DIR.exists():
        return refs
    for filepath in REFERENCE_DIR.glob("*.json"):
        try:
            with open(filepath) as f:
                record = json.load(f)
            refs[record["name"]] = {
                "col_hint": record["col_hint"],
                "distribution": record["distribution"],
                "custom": True,
            }
        except Exception:
            continue
    return refs


def get_reference_for_column(col_name: str, all_refs: dict = None) -> dict:
    """Find the best matching reference for a given column name."""
    if all_refs is None:
        all_refs = load_all_references()
    col_lower = col_name.lower()
    for name, ref in all_refs.items():
        hint = ref.get("col_hint", "")
        if hint and (hint in col_lower or col_lower in hint):
            return ref["distribution"]
    return {}