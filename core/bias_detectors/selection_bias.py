import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from core.bias_result import BiasResult


class SelectionBiasDetector:
    def __init__(self, config: dict):
        self.mcar_p_threshold = config.get("mcar_p_threshold", 0.05)

    def detect(self, df: pd.DataFrame, profile: dict) -> BiasResult:
        evidence = {}
        flagged = []

        try:
            cols_with_nulls = [
                col for col, info in profile["columns"].items()
                if info.get("null_rate", 0) > 0.01
            ]
            categorical_cols = [
                col for col, info in profile["columns"].items()
                if info.get("kind") == "categorical" and df[col].nunique() < 20
            ]

            # 1. Correlated missingness
            for null_col in cols_with_nulls[:10]:  # cap to avoid slowness
                for group_col in categorical_cols[:10]:
                    if null_col == group_col:
                        continue
                    try:
                        missing_mask = df[null_col].isna()
                        # skip if all missing or none missing
                        if missing_mask.all() or not missing_mask.any():
                            continue
                        contingency = pd.crosstab(missing_mask, df[group_col])
                        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                            continue
                        chi2, p, _, _ = chi2_contingency(contingency)
                        if p < self.mcar_p_threshold:
                            key = f"{null_col}_missing_vs_{group_col}"
                            evidence[key] = {
                                "chi2": round(float(chi2), 4),
                                "p_value": round(float(p), 4),
                                "interpretation": (
                                    f"Missingness in '{null_col}' correlates "
                                    f"with '{group_col}' — data is NOT missing at random."
                                )
                            }
                            flagged.append(null_col)
                    except Exception:
                        continue

            # 2. Zero variance columns
            for col, info in profile["columns"].items():
                if info.get("kind") != "numeric":
                    continue
                col_min = info.get("min", 0)
                col_max = info.get("max", 0)
                if col_min is not None and col_max is not None:
                    if col_min == col_max:
                        evidence[f"{col}_zero_variance"] = {
                            "interpretation": f"'{col}' has zero variance — constant value."
                        }
                        flagged.append(col)

                # 3. Extreme skew
                skewness = info.get("skewness")
                if skewness is not None and not np.isnan(float(skewness)) and abs(float(skewness)) > 3:
                    evidence[f"{col}_high_skew"] = {
                        "skewness": round(float(skewness), 4),
                        "interpretation": (
                            f"'{col}' is heavily skewed ({skewness:.2f}), "
                            "suggesting a non-representative slice of data."
                        )
                    }
                    flagged.append(col)

        except Exception as e:
            return BiasResult(
                bias_type="selection_bias",
                severity="unknown",
                affected_columns=[],
                evidence={"error": str(e)},
                recommendation="Detector encountered an error — check logs.",
                detected=False,
            )

        flagged = list(set(flagged))
        detected = len(flagged) > 0
        severity = "high" if len(flagged) > 3 else "medium" if flagged else "low"

        return BiasResult(
            bias_type="selection_bias",
            severity=severity,
            affected_columns=flagged,
            evidence=evidence,
            recommendation=(
                "Missing values are non-random (MAR/MNAR). Dataset may not represent "
                "the full population. Review data collection pipelines."
            ) if detected else "No significant selection bias detected.",
            detected=detected,
        )