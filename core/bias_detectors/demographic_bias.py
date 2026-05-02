import pandas as pd
import numpy as np
from scipy.stats import chisquare
from core.bias_result import BiasResult

DEFAULT_REFERENCES = {
    "gender": {"male": 0.495, "female": 0.495, "other": 0.01},
    "sex":    {"male": 0.495, "female": 0.495, "other": 0.01},
    "race": {
        "white": 0.595, "black": 0.134, "hispanic": 0.186,
        "asian": 0.059, "other": 0.026,
    },
}

ALIAS_MAP = {
    "african-american": "black", "african american": "black",
    "caucasian": "white", "european american": "white",
    "latino": "hispanic", "latina": "hispanic",
    "asian-pacific islander": "asian",
    "native american": "other", "middle eastern": "other",
    "m": "male", "f": "female",
    "male": "male", "female": "female",
}


def normalize_value(val: str) -> str:
    return ALIAS_MAP.get(val.lower().strip(), val.lower().strip())


class DemographicBiasDetector:
    def __init__(self, config: dict):
        self.threshold_p = config.get("demographic_p_threshold", 0.05)
        self.underrep_threshold = config.get("underrep_threshold", 0.20)

        # Merge default + custom references
        self.references = dict(DEFAULT_REFERENCES)
        custom_refs = config.get("references", {})
        for ref_name, ref_data in custom_refs.items():
            if isinstance(ref_data, dict) and "col_hint" in ref_data:
                hint = ref_data["col_hint"].lower()
                self.references[hint] = ref_data["distribution"]

    def detect(self, df: pd.DataFrame, profile: dict) -> BiasResult:
        protected_cols = profile.get("suspected_protected_attrs", [])
        if not protected_cols:
            return BiasResult(
                bias_type="demographic_disparity",
                severity="low",
                affected_columns=[],
                evidence={"message": "No protected attribute columns detected."},
                recommendation="Manually specify protected attribute columns if applicable.",
                detected=False,
            )

        findings = {}
        flagged_cols = []

        for col in protected_cols:
            if col not in df.columns:
                continue
            if profile["columns"].get(col, {}).get("kind") != "categorical":
                continue

            normalized_series = df[col].astype(str).apply(normalize_value)
            observed_dist = normalized_series.value_counts(normalize=True)

            # Find matching reference
            ref_dist = None
            for hint, dist in self.references.items():
                if hint in col.lower() or col.lower() in hint:
                    ref_dist = dist
                    break

            finding = {
                "observed": {str(k): round(float(v), 4) for k, v in observed_dist.items()},
                "original_values": {str(k): int(v) for k, v in df[col].value_counts().head(10).items()},
            }

            if ref_dist:
                all_keys = list(ref_dist.keys())
                obs_aligned = [float(observed_dist.get(k, 0)) for k in all_keys]
                exp_aligned = [ref_dist[k] for k in all_keys]
                exp_total = sum(exp_aligned)
                exp_aligned = [e / exp_total for e in exp_aligned]
                n = len(df)

                try:
                    stat, p = chisquare(
                        f_obs=[max(o * n, 0.001) for o in obs_aligned],
                        f_exp=[max(e * n, 0.001) for e in exp_aligned],
                    )
                    gaps = {k: round(ref_dist[k] - float(observed_dist.get(k, 0)), 4)
                            for k in all_keys}
                    finding.update({
                        "chi2_stat": round(float(stat), 4),
                        "p_value": round(float(p), 4),
                        "reference_distribution": ref_dist,
                        "representation_gaps": gaps,
                        "significant": bool(p < self.threshold_p),
                    })
                    max_gap = max(gaps.values())
                    if p < self.threshold_p and max_gap > self.underrep_threshold:
                        flagged_cols.append(col)
                except Exception as e:
                    finding["error"] = str(e)
            else:
                finding["note"] = "No reference distribution found. Reporting observed only."
                if len(observed_dist) > 0 and float(observed_dist.min()) < self.underrep_threshold:
                    flagged_cols.append(col)

            findings[col] = finding

        detected = len(flagged_cols) > 0
        severity = "high" if len(flagged_cols) > 1 else "medium" if flagged_cols else "low"

        return BiasResult(
            bias_type="demographic_disparity",
            severity=severity,
            affected_columns=flagged_cols,
            evidence=findings,
            recommendation=(
                "Protected attribute groups are significantly underrepresented. "
                "Consider collecting more data or using the synthetic balancer."
            ) if detected else "No significant demographic disparity detected.",
            detected=detected,
        )