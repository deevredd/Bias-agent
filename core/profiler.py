import pandas as pd
import numpy as np


PROTECTED_ATTR_HINTS = [
    "gender", "sex", "race", "ethnicity", "age", "religion",
    "nationality", "disability", "marital", "zip", "income"
]


def safe(val):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if pd.isna(val) if not isinstance(val, (list, dict, str)) else False:
        return None
    return val


class DataProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.profile = {}

    def run(self) -> dict:
        self.profile = {
            "shape": {"rows": int(len(self.df)), "cols": int(len(self.df.columns))},
            "columns": {}
        }
        for col in self.df.columns:
            try:
                self.profile["columns"][col] = self._profile_column(col)
            except Exception as e:
                self.profile["columns"][col] = {"error": str(e)}

        self.profile["suspected_protected_attrs"] = self._detect_protected_attrs()
        self.profile["suspected_datetime_cols"] = self._detect_datetime_cols()
        self.profile["suspected_target_cols"] = self._detect_target_cols()
        return self.profile

    def _profile_column(self, col: str) -> dict:
        series = self.df[col]
        base = {
            "dtype": str(series.dtype),
            "null_rate": safe(round(float(series.isna().mean()), 4)),
            "unique_count": int(series.nunique()),
            "unique_rate": safe(round(float(series.nunique() / max(len(series), 1)), 4)),
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            base.update({
                "kind": "numeric",
                "mean": safe(round(float(desc["mean"]), 4)),
                "std": safe(round(float(desc["std"]), 4)),
                "min": safe(float(desc["min"])),
                "25%": safe(float(desc["25%"])),
                "50%": safe(float(desc["50%"])),
                "75%": safe(float(desc["75%"])),
                "max": safe(float(desc["max"])),
                "skewness": safe(round(float(series.skew()), 4)),
                "kurtosis": safe(round(float(series.kurt()), 4)),
            })
        elif pd.api.types.is_datetime64_any_dtype(series):
            base.update({
                "kind": "datetime",
                "min": str(series.min()),
                "max": str(series.max()),
            })
        else:
            freq = series.value_counts(normalize=True)
            base.update({
                "kind": "categorical",
                "top_values": {str(k): safe(float(v)) for k, v in freq.head(10).items()},
            })
        return base

    def _detect_protected_attrs(self) -> list:
        found = []
        for col in self.df.columns:
            if any(hint in col.lower() for hint in PROTECTED_ATTR_HINTS):
                found.append(col)
        return found

    def _detect_datetime_cols(self) -> list:
        found = []
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                found.append(col)
            elif any(x in col.lower() for x in ["date", "time", "year"]):
                try:
                    pd.to_datetime(self.df[col], errors="raise")
                    found.append(col)
                except Exception:
                    pass
        return found

    def _detect_target_cols(self) -> list:
        target_hints = [
            "label", "target", "outcome", "result", "status",
            "churn", "default", "promoted", "survived", "approved",
            "recid", "score", "risk"
        ]
        found = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(hint in col_lower for hint in target_hints):
                found.append(col)
            elif self.df[col].nunique() == 2:
                found.append(col)
        return list(set(found))