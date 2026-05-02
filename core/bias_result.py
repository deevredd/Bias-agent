from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np


def make_serializable(obj):
    """Recursively convert numpy/pandas types to native Python."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


@dataclass
class BiasResult:
    bias_type: str
    severity: str
    affected_columns: List[str]
    evidence: Dict[str, Any]
    recommendation: str
    detected: bool = True

    def to_dict(self):
        return make_serializable({
            "bias_type": self.bias_type,
            "severity": self.severity,
            "affected_columns": self.affected_columns,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "detected": self.detected,
        })