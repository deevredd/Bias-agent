import pandas as pd
import numpy as np
from core.bias_result import BiasResult


class SurvivorshipBiasDetector:
    def __init__(self, config: dict):
        self.dominance_threshold = config.get("survivorship_dominance_threshold", 0.90)

    def detect(self, df: pd.DataFrame, profile: dict) -> BiasResult:
        evidence = {}
        flagged = []

        suspected_targets = profile.get("suspected_target_cols", [])

        for col in suspected_targets:
            if col not in df.columns:
                continue
            col_info = profile["columns"].get(col, {})

            # Binary column check
            if df[col].nunique() == 2:
                value_counts = df[col].value_counts(normalize=True)
                dominant_class = value_counts.index[0]
                dominant_rate = value_counts.iloc[0]

                if dominant_rate >= self.dominance_threshold:
                    evidence[col] = {
                        "dominant_class": str(dominant_class),
                        "dominant_rate": round(dominant_rate, 4),
                        "interpretation": (
                            f"'{col}' is {dominant_rate*100:.1f}% '{dominant_class}'. "
                            "If this is an outcome column, the dataset may only contain "
                            "survivors/successes, hiding failures from analysis."
                        )
                    }
                    flagged.append(col)

            # Categorical with very low failure/negative class rate
            elif col_info.get("kind") == "categorical":
                top_values = col_info.get("top_values", {})
                if top_values:
                    top_rate = list(top_values.values())[0]
                    if top_rate >= self.dominance_threshold:
                        evidence[col] = {
                            "top_value": list(top_values.keys())[0],
                            "rate": round(top_rate, 4),
                            "interpretation": (
                                f"'{col}' is dominated by one outcome "
                                f"({top_rate*100:.1f}%). Possible survivorship bias."
                            )
                        }
                        flagged.append(col)

        detected = len(flagged) > 0
        severity = "high" if len(flagged) >= 2 else "medium" if flagged else "low"

        return BiasResult(
            bias_type="survivorship_bias",
            severity=severity,
            affected_columns=flagged,
            evidence=evidence,
            recommendation=(
                "Outcome columns are heavily imbalanced toward positive/success cases. "
                "Ensure your dataset includes failed, churned, or excluded cases. "
                "If not available, document this limitation clearly."
            ) if detected else "No significant survivorship bias detected.",
            detected=detected,
        )