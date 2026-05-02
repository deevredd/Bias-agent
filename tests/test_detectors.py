import pandas as pd
import numpy as np
import pytest
from core.profiler import DataProfiler
from core.bias_detectors import run_all_detectors


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "age": np.random.randint(18, 65, n),
        "income": np.random.normal(50000, 15000, n),
        "gender": np.random.choice(["male", "female"], n, p=[0.85, 0.15]),
        "race": np.random.choice(["white", "black", "hispanic", "asian"], n, p=[0.80, 0.08, 0.08, 0.04]),
        "outcome": np.random.choice([0, 1], n, p=[0.05, 0.95]),
        "date": pd.date_range("2018-01-01", periods=n, freq="D"),
    })


def test_profiler_runs(sample_df):
    profile = DataProfiler(sample_df).run()
    assert "columns" in profile
    assert "shape" in profile
    assert profile["shape"]["rows"] == 500


def test_detectors_return_list(sample_df):
    profile = DataProfiler(sample_df).run()
    results = run_all_detectors(sample_df, profile)
    assert isinstance(results, list)
    assert len(results) == 4


def test_demographic_bias_detected(sample_df):
    profile = DataProfiler(sample_df).run()
    results = run_all_detectors(sample_df, profile)
    demographic = next(r for r in results if r["bias_type"] == "demographic_disparity")
    assert demographic["detected"] is True


def test_survivorship_bias_detected(sample_df):
    profile = DataProfiler(sample_df).run()
    results = run_all_detectors(sample_df, profile)
    surv = next(r for r in results if r["bias_type"] == "survivorship_bias")
    assert surv["detected"] is True


def test_temporal_bias_detected(sample_df):
    profile = DataProfiler(sample_df).run()
    results = run_all_detectors(sample_df, profile)
    temporal = next(r for r in results if r["bias_type"] == "temporal_bias")
    assert "temporal_bias" == temporal["bias_type"]