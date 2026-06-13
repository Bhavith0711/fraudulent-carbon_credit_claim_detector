from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_to_0_100(series: pd.Series) -> pd.Series:
    min_v = float(series.min())
    max_v = float(series.max())
    if np.isclose(max_v - min_v, 0.0):
        return pd.Series(np.full(len(series), 50.0), index=series.index)
    return (series - min_v) / (max_v - min_v) * 100


def assign_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    abs_gap = np.abs(out["reduction_gap_tco2e"])
    gap_component = _normalize_to_0_100(abs_gap)
    anomaly_component = _normalize_to_0_100(out["anomaly_score"])
    energy_component = _normalize_to_0_100(np.abs(out["energy_change_pct"]))

    # Larger values indicate more suspicious behavior.
    suspicion = 0.45 * gap_component + 0.4 * anomaly_component + 0.15 * energy_component
    out["credibility_score"] = (100 - suspicion).clip(lower=0, upper=100).round(2)

    conditions = [
        out["credibility_score"] >= 75,
        (out["credibility_score"] >= 45) & (out["credibility_score"] < 75),
        out["credibility_score"] < 45,
    ]
    choices = ["LOW", "MEDIUM", "HIGH"]
    out["risk_level"] = np.select(conditions, choices, default="MEDIUM")

    out["recommendation"] = out["risk_level"].map(
        {
            "LOW": "Approve claim",
            "MEDIUM": "Request additional evidence",
            "HIGH": "Escalate for physical audit",
        }
    )

    return out
