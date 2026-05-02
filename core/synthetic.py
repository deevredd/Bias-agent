import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


def balance_with_smote(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()

    # Encode target if it's not numeric
    y_raw = df[target_col].astype(str)
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw)

    # Drop non-numeric columns for SMOTE, keep only numeric features
    feature_df = df.drop(columns=[target_col])

    # Encode categorical columns numerically
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))
        encoders[col] = le

    # Fill nulls
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    # Drop any remaining non-numeric columns
    feature_df = feature_df.select_dtypes(include=[np.number])

    if feature_df.shape[1] == 0:
        raise ValueError("No numeric features available for SMOTE after encoding.")

    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        raise ValueError(f"Target column '{target_col}' must have at least 2 classes.")

    min_count = counts.min()
    k_neighbors = min(5, min_count - 1)
    if k_neighbors < 1:
        raise ValueError(
            f"Not enough samples in minority class ({min_count}) to run SMOTE. "
            "Need at least 2 samples in the smallest class."
        )

    try:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(feature_df, y)
    except Exception as e:
        raise ValueError(f"SMOTE failed: {e}")

    balanced_df = pd.DataFrame(X_res, columns=feature_df.columns)
    # Decode target back to original labels
    balanced_df[target_col] = le_target.inverse_transform(y_res)
    balanced_df["is_synthetic"] = balanced_df.index >= len(df)
    return balanced_df


def balance_demographic_group(
    df: pd.DataFrame,
    group_col: str,
    target_group: str,
    target_count: int,
) -> pd.DataFrame:
    subset = df[df[group_col] == target_group].copy()
    current_count = len(subset)
    needed = target_count - current_count

    if needed <= 0:
        result = df.copy()
        result["is_synthetic"] = False
        return result

    try:
        from ctgan import CTGAN
        model = CTGAN(epochs=100, verbose=False)
        model.fit(subset)
        synthetic = model.sample(needed)
        synthetic[group_col] = target_group
        synthetic["is_synthetic"] = True
    except ImportError:
        # Fallback: bootstrap with jitter
        sampled = subset.sample(n=needed, replace=True, random_state=42).copy()
        numeric_cols = sampled.select_dtypes(include=[np.number]).columns
        noise = np.random.normal(0, 0.01, sampled[numeric_cols].shape)
        sampled[numeric_cols] = sampled[numeric_cols] + noise
        sampled["is_synthetic"] = True
        synthetic = sampled

    original = df.copy()
    original["is_synthetic"] = False
    return pd.concat([original, synthetic], ignore_index=True)