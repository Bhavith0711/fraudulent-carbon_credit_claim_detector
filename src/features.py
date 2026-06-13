from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    epsilon = 1e-6

    features["actual_reduction_tco2e"] = (
        features["historical_avg_emissions_tco2e"] - features["current_emissions_tco2e"]
    )
    features["expected_reduction_tco2e"] = features["actual_reduction_tco2e"].clip(lower=0)
    features["reduction_gap_tco2e"] = (
        features["claimed_reduction_tco2e"] - features["expected_reduction_tco2e"]
    )
    features["reduction_gap_ratio"] = features["reduction_gap_tco2e"] / (
        np.abs(features["expected_reduction_tco2e"]) + epsilon
    )
    features["energy_change_pct"] = (
        (features["current_energy_mwh"] - features["historical_avg_energy_mwh"])
        / (features["historical_avg_energy_mwh"] + epsilon)
        * 100
    )
    features["emissions_change_pct"] = (
        (features["current_emissions_tco2e"] - features["historical_avg_emissions_tco2e"])
        / (features["historical_avg_emissions_tco2e"] + epsilon)
        * 100
    )

    return features


def model_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "claimed_reduction_tco2e",
            "expected_reduction_tco2e",
            "reduction_gap_tco2e",
            "reduction_gap_ratio",
            "energy_change_pct",
            "emissions_change_pct",
        ]
    ]
