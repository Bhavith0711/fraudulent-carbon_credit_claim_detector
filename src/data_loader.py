from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "claim_id",
    "organization",
    "claimed_reduction_tco2e",
    "historical_avg_emissions_tco2e",
    "current_emissions_tco2e",
    "historical_avg_energy_mwh",
    "current_energy_mwh",
]


DATASET_OPTIONS = {
    "sample": "data/sample_claims.csv",
    "sample_fraud": "data/sample_claims_fraud.csv",
    "large": "data/large_claims.csv",
    "large_fraud": "data/large_claims_fraud.csv",
    "large_1000": "data/large_claims_1000.csv",
    "large_fraud_1000": "data/large_claims_fraud_1000.csv",
}


def get_dataset_options() -> list[str]:
    return sorted(DATASET_OPTIONS.keys())


def resolve_dataset_path(dataset: str | None, path: str | None) -> str:
    if dataset and path:
        raise ValueError("Provide either a dataset option or a custom path, not both.")
    if dataset:
        if dataset not in DATASET_OPTIONS:
            raise ValueError(
                f"Unknown dataset option '{dataset}'. Valid options: {', '.join(get_dataset_options())}"
            )
        return DATASET_OPTIONS[dataset]
    if path:
        return path
    raise ValueError("Missing dataset source. Provide --dataset or --input.")


def load_claims(path: str | None = None, dataset: str | None = None) -> pd.DataFrame:
    source_path = resolve_dataset_path(dataset=dataset, path=path)
    if not Path(source_path).exists():
        raise ValueError(f"Input file not found: {source_path}")
    df = pd.read_csv(source_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [col for col in REQUIRED_COLUMNS if col not in ["claim_id", "organization"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if df[numeric_cols].isna().any().any():
        bad_rows = df[df[numeric_cols].isna().any(axis=1)]
        raise ValueError(f"Found invalid numeric values in rows: {bad_rows.index.tolist()}")

    return df.copy()
