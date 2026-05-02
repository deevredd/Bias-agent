import pandas as pd
import numpy as np


def explain_bias_results(
    df: pd.DataFrame,
    bias_results: list,
    profile: dict,
) -> list:
    """
    Attach a human-readable explainability block to each bias result.
    Mutates and returns the bias_results list.
    """
    n = len(df)
    for result in bias_results:
        try:
            result["explainability"] = _explain_one(df, result, profile, n)
        except Exception as e:
            result["explainability"] = {
                "summary": "Could not generate explanation.",
                "examples": [],
                "error": str(e),
            }
    return bias_results


def _explain_one(df, result, profile, n) -> dict:
    bias_type = result["bias_type"]

    if bias_type == "demographic_disparity":
        return _explain_demographic(df, result, n)
    elif bias_type == "selection_bias":
        return _explain_selection(df, result, n)
    elif bias_type == "survivorship_bias":
        return _explain_survivorship(df, result, n)
    elif bias_type == "temporal_bias":
        return _explain_temporal(df, result, profile)
    return {"summary": "", "examples": []}


def _explain_demographic(df, result, n) -> dict:
    evidence = result.get("evidence", {})
    examples = []
    summaries = []

    for col, findings in evidence.items():
        if not isinstance(findings, dict) or "observed" not in findings:
            continue

        observed = findings["observed"]
        ref = findings.get("reference_distribution", {})
        gaps = findings.get("representation_gaps", {})

        # Find the most underrepresented group
        if gaps:
            worst_group = max(gaps, key=lambda k: gaps[k])
            worst_gap = gaps[worst_group]
            if worst_gap > 0:
                observed_rate = observed.get(worst_group, 0)
                ref_rate = ref.get(worst_group, 0)
                observed_count = int(observed_rate * n)
                expected_count = int(ref_rate * n)

                summaries.append(
                    f"The '{col}' column shows significant group imbalance. "
                    f"'{worst_group.title()}' makes up {observed_rate*100:.1f}% of this dataset "
                    f"but {ref_rate*100:.1f}% of the reference population — "
                    f"a gap of {worst_gap*100:.1f} percentage points."
                )
                examples.append(
                    f"A model trained on this data will have seen only {observed_count:,} "
                    f"'{worst_group}' examples vs {expected_count:,} expected. "
                    f"This can cause the model to perform {worst_gap*100:.0f}% worse "
                    f"on underrepresented groups."
                )

                # Find most overrepresented group
                best_group = min(gaps, key=lambda k: gaps[k])
                if gaps[best_group] < -0.05:
                    over_rate = observed.get(best_group, 0)
                    examples.append(
                        f"Conversely, '{best_group.title()}' is overrepresented at "
                        f"{over_rate*100:.1f}% vs {ref.get(best_group,0)*100:.1f}% reference. "
                        f"The model will be biased toward this group's patterns."
                    )

    return {
        "summary": " ".join(summaries) if summaries else "Demographic groups are not equally represented.",
        "examples": examples,
    }


def _explain_selection(df, result, n) -> dict:
    evidence = result.get("evidence", {})
    examples = []
    skewed_cols = []
    missing_pairs = []

    for key, val in evidence.items():
        if not isinstance(val, dict):
            continue
        if "missing_vs" in key:
            parts = key.split("_missing_vs_")
            if len(parts) == 2:
                missing_pairs.append((parts[0], parts[1]))
        if "high_skew" in key:
            col = key.replace("_high_skew", "")
            skewed_cols.append((col, val.get("skewness", 0)))

    if missing_pairs:
        col, group = missing_pairs[0]
        null_rate = df[col].isna().mean()
        null_count = df[col].isna().sum()
        examples.append(
            f"'{col}' is missing {null_count:,} values ({null_rate*100:.1f}%), "
            f"and this missingness clusters around specific values of '{group}'. "
            f"This means certain groups are systematically excluded from the data."
        )

    if skewed_cols:
        col, skew = skewed_cols[0]
        examples.append(
            f"'{col}' has a skewness of {skew:.1f} — far above the normal range of ±1. "
            f"This suggests only a narrow slice of the true population was captured, "
            f"not a representative sample."
        )

    if not result["detected"]:
        return {"summary": "No selection bias detected in this dataset.", "examples": []}

    return {
        "summary": (
            f"This dataset shows signs of non-random data collection. "
            f"{len(missing_pairs)} column(s) have missingness that correlates with "
            f"group membership, meaning some groups are systematically undersampled."
        ),
        "examples": examples,
    }


def _explain_survivorship(df, result, n) -> dict:
    evidence = result.get("evidence", {})
    examples = []

    for col, val in evidence.items():
        if not isinstance(val, dict):
            continue
        dominant = val.get("dominant_class")
        rate = val.get("dominant_rate", 0)
        minority_rate = 1 - rate
        minority_count = int(minority_rate * n)

        examples.append(
            f"'{col}' is {rate*100:.1f}% '{dominant}'. Only {minority_count:,} records "
            f"({minority_rate*100:.1f}%) represent the non-dominant outcome. "
            f"A model trained here will rarely learn from failure cases."
        )
        examples.append(
            f"Real-world example: if this is a loan dataset and '{dominant}' means 'repaid', "
            f"the model never truly learns what causes defaults — it only sees successful cases."
        )

    if not result["detected"]:
        return {"summary": "No survivorship bias detected.", "examples": []}

    return {
        "summary": (
            "The dataset disproportionately represents 'successful' or 'surviving' cases. "
            "Failed, churned, or excluded outcomes are severely underrepresented, "
            "which will cause models to be overconfident and miss real-world failure patterns."
        ),
        "examples": examples,
    }


def _explain_temporal(df, result, profile) -> dict:
    evidence = result.get("evidence", {})
    drift_items = {k: v for k, v in evidence.items() if "_drift" in k}
    examples = []

    if drift_items:
        # Most drifted column
        worst = max(drift_items.items(), key=lambda x: x[1].get("ks_stat", 0))
        col = worst[0].replace("_drift", "")
        val = worst[1]
        early_mean = val.get("early_mean", 0)
        late_mean = val.get("late_mean", 0)
        direction = "increased" if late_mean > early_mean else "decreased"

        examples.append(
            f"'{col}' has {direction} from a mean of {early_mean:.2f} (early period) "
            f"to {late_mean:.2f} (recent period). A model trained on the full dataset "
            f"treats these as the same distribution — it will underperform on recent data."
        )
        examples.append(
            f"Practical impact: if you train on historical data and deploy today, "
            f"predictions for '{col}' will be based on outdated patterns. "
            f"Use time-aware train/test splits to avoid this."
        )

    sparse = evidence.get("sparse_time_periods", {})
    if sparse and sparse.get("periods"):
        periods = sparse["periods"][:3]
        examples.append(
            f"Data is extremely sparse during: {', '.join(periods)}. "
            f"Models will effectively ignore these time periods during training, "
            f"creating blind spots for anything that happened then."
        )

    if not result["detected"]:
        return {"summary": "No temporal bias detected.", "examples": []}

    return {
        "summary": (
            f"Feature distributions shift significantly across time in this dataset. "
            f"{len(drift_items)} column(s) show statistically significant drift "
            f"between early and recent time periods."
        ),
        "examples": examples,
    }