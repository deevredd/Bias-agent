from core.bias_detectors.demographic_bias import DemographicBiasDetector
from core.bias_detectors.selection_bias import SelectionBiasDetector
from core.bias_detectors.survivorship_bias import SurvivorshipBiasDetector
from core.bias_detectors.temporal_bias import TemporalBiasDetector


def run_all_detectors(df, profile, config=None, references=None) -> list:
    config = config or {}
    if references:
        config["references"] = references

    detectors = [
        DemographicBiasDetector(config),
        SelectionBiasDetector(config),
        SurvivorshipBiasDetector(config),
        TemporalBiasDetector(config),
    ]
    results = []
    for detector in detectors:
        try:
            result = detector.detect(df, profile)
            if result:
                results.append(result.to_dict())
        except Exception as e:
            results.append({
                "bias_type": detector.__class__.__name__,
                "severity": "unknown",
                "affected_columns": [],
                "evidence": {"error": str(e)},
                "recommendation": "Detector failed — check logs.",
                "explainability": {"summary": "", "examples": []},
                "detected": False,
            })
    return results