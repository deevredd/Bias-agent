import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from core.bias_result import BiasResult


class TemporalBiasDetector:
    def __init__(self, config: dict):
        self.ks_p_threshold = config.get("temporal_ks_p_threshold", 0.05)
        self.density_gap_threshold = config.get("density_gap_threshold", 0.5)

    def detect(self, df: pd.DataFrame, profile: dict) -> BiasResult:
        datetime_cols = profile.get("suspected_datetime_cols", [])
        evidence = {}
        flagged = []

        if not datetime_cols:
            return BiasResult(
                bias_type="temporal_bias",
                severity="low",
                affected_columns=[],
                evidence={"message": "No datetime columns detected."},
                recommendation="If your dataset has time-based data, ensure a datetime column is included.",
                detected=False,
            )

        time_col = datetime_cols[0]
        try:
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col]).sort_values(time_col)
        except Exception as e:
            return BiasResult(
                bias_type="temporal_bias",
                severity="low",
                affected_columns=[],
                evidence={"error": str(e)},
                recommendation="Could not parse datetime column.",
                detected=False,
            )

        midpoint = len(df) // 2
        early = df.iloc[:midpoint]
        late = df.iloc[midpoint:]

        numeric_cols = [
            col for col, info in profile["columns"].items()
            if info.get("kind") == "numeric" and col != time_col
        ]

        drifted_cols = []
        for col in numeric_cols:
            try:
                stat, p = ks_2samp(
                    early[col].dropna().values,
                    late[col].dropna().values
                )
                if p < self.ks_p_threshold:
                    drifted_cols.append(col)
                    evidence[f"{col}_drift"] = {
                        "ks_stat": round(stat, 4),
                        "p_value": round(p, 4),
                        "early_mean": round(early[col].mean(), 4),
                        "late_mean": round(late[col].mean(), 4),
                        "interpretation": (
                            f"'{col}' distribution shifted significantly over time "
                            f"(KS={stat:.3f}, p={p:.4f})."
                        )
                    }
            except Exception:
                continue

        # Density gap detection — are there time periods with very few records?
        df["_period"] = df[time_col].dt.to_period("M")
        period_counts = df["_period"].value_counts().sort_index()
        mean_count = period_counts.mean()
        sparse_periods = period_counts[period_counts < mean_count * self.density_gap_threshold]

        if not sparse_periods.empty:
            evidence["sparse_time_periods"] = {
                "periods": [str(p) for p in sparse_periods.index.tolist()],
                "interpretation": (
                    "Some time periods have far fewer records than average. "
                    "Models trained on this data may underweight these periods."
                )
            }
            flagged.append(time_col)

        flagged.extend(drifted_cols)
        flagged = list(set(flagged))
        detected = len(flagged) > 0
        severity = "high" if len(drifted_cols) > 3 else "medium" if flagged else "low"

        return BiasResult(
            bias_type="temporal_bias",
            severity=severity,
            affected_columns=flagged,
            evidence=evidence,
            recommendation=(
                "Feature distributions shift significantly over time. "
                "Use time-aware train/test splits. Retrain models regularly. "
                "Investigate sparse time periods for data collection issues."
            ) if detected else "No significant temporal bias detected.",
            detected=detected,
        )