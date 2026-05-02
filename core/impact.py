import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


def _encode_df(df: pd.DataFrame, target_col: str, drop_cols: list = None):
    drop_cols = drop_cols or []
    drop_cols = [c for c in drop_cols if c in df.columns]
    extra = [c for c in ["is_synthetic"] if c in df.columns]
    feature_df = df.drop(columns=[target_col] + drop_cols + extra).copy()

    for col in feature_df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))

    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    feature_df = feature_df.select_dtypes(include=[np.number])
    return feature_df


def simulate_impact(
    original_df: pd.DataFrame,
    balanced_df: pd.DataFrame,
    target_col: str,
    protected_col: str = None,
) -> dict:
    results = {}

    for label, df in [("biased", original_df), ("balanced", balanced_df)]:
        try:
            df = df.copy()

            # Encode target
            le_target = LabelEncoder()
            y = le_target.fit_transform(df[target_col].astype(str))

            # Save protected column before dropping
            protected_series = None
            if protected_col and protected_col in df.columns:
                protected_series = df[protected_col].reset_index(drop=True)

            X = _encode_df(df, target_col, drop_cols=[protected_col] if protected_col else [])

            if X.shape[1] == 0:
                results[label] = {"error": "No numeric features available after encoding."}
                continue

            X = X.reset_index(drop=True)
            y_series = pd.Series(y)

            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X, y_series, X.index, test_size=0.25, random_state=42, stratify=y_series
            )

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            overall = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            }

            group_metrics = {}
            if protected_series is not None:
                test_protected = protected_series.iloc[idx_test].reset_index(drop=True)
                y_test_reset = y_test.reset_index(drop=True)
                y_pred_series = pd.Series(y_pred)

                for group in test_protected.unique():
                    mask = (test_protected == group).values
                    if mask.sum() < 5:
                        continue
                    group_metrics[str(group)] = {
                        "count": int(mask.sum()),
                        "accuracy": round(float(accuracy_score(y_test_reset[mask], y_pred_series[mask])), 4),
                        "precision": round(float(precision_score(y_test_reset[mask], y_pred_series[mask], average="weighted", zero_division=0)), 4),
                        "recall": round(float(recall_score(y_test_reset[mask], y_pred_series[mask], average="weighted", zero_division=0)), 4),
                        "f1": round(float(f1_score(y_test_reset[mask], y_pred_series[mask], average="weighted", zero_division=0)), 4),
                    }

            results[label] = {
                "overall": overall,
                "per_group": group_metrics,
            }

        except Exception as e:
            results[label] = {"error": str(e)}

    return results