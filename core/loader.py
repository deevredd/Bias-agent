import pandas as pd
from pathlib import Path


def load_dataset(source) -> pd.DataFrame:
    """
    Accept a file path (str/Path) or file-like object.
    Supports CSV, Parquet, JSON.
    """
    if isinstance(source, (str, Path)):
        source = Path(source)
        suffix = source.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(source)
        elif suffix == ".parquet":
            return pd.read_parquet(source)
        elif suffix == ".json":
            return pd.read_json(source)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    else:
        # file-like object — try CSV first
        try:
            return pd.read_csv(source)
        except Exception:
            raise ValueError("Could not parse uploaded file as CSV.")