from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _build_row(claim_id: str, org: str, rng: np.random.Generator, fraud: bool) -> dict:
    historical_emissions = float(rng.uniform(4200, 16000))
    historical_energy = float(rng.uniform(2100, 7600))

    # Normal operations: emissions and energy generally decrease together.
    if not fraud:
        emission_drop_ratio = float(rng.uniform(0.05, 0.18))
        energy_drop_ratio = float(rng.uniform(0.03, 0.15))
        current_emissions = historical_emissions * (1 - emission_drop_ratio)
        current_energy = historical_energy * (1 - energy_drop_ratio)
        actual_reduction = historical_emissions - current_emissions
        claimed_reduction = actual_reduction * float(rng.uniform(0.90, 1.10))
    else:
        # Fraud patterns: inflated claims, weak emissions reduction, and often increased energy.
        emission_drop_ratio = float(rng.uniform(0.00, 0.05))
        energy_change_ratio = float(rng.uniform(0.02, 0.25))
        current_emissions = historical_emissions * (1 - emission_drop_ratio)
        current_energy = historical_energy * (1 + energy_change_ratio)
        actual_reduction = historical_emissions - current_emissions
        claimed_reduction = max(
            actual_reduction * float(rng.uniform(2.2, 5.0)),
            historical_emissions * float(rng.uniform(0.18, 0.35)),
        )

    return {
        "claim_id": claim_id,
        "organization": org,
        "claimed_reduction_tco2e": round(max(claimed_reduction, 0.0), 2),
        "historical_avg_emissions_tco2e": round(historical_emissions, 2),
        "current_emissions_tco2e": round(max(current_emissions, 0.0), 2),
        "historical_avg_energy_mwh": round(historical_energy, 2),
        "current_energy_mwh": round(max(current_energy, 0.0), 2),
    }


def generate_dataset(rows: int, fraud_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    fraud_count = int(rows * fraud_ratio)
    fraud_indices = set(rng.choice(rows, size=fraud_count, replace=False).tolist())

    org_prefixes = [
        "Green", "Blue", "Terra", "Eco", "Nova", "Sun", "Hydro", "Urban", "Cedar", "Aqua"
    ]
    org_suffixes = [
        "Metals", "Cement", "Foods", "Textiles", "Chem", "Power", "Alloys", "Materials", "Systems", "Labs"
    ]

    rows_normal: list[dict] = []
    rows_with_fraud: list[dict] = []

    for i in range(rows):
        claim_id = f"C{i + 1:04d}"
        org = f"{rng.choice(org_prefixes)}{rng.choice(org_suffixes)} {rng.integers(1, 999)}"
        is_fraud = i in fraud_indices
        row = _build_row(claim_id, org, rng, fraud=is_fraud)

        rows_with_fraud.append(row)
        if not is_fraud:
            rows_normal.append(row)

    normal_df = pd.DataFrame(rows_normal)
    fraud_df = pd.DataFrame(rows_with_fraud)
    return normal_df, fraud_df


def append_c099_demo_claim(fraud_df: pd.DataFrame) -> pd.DataFrame:
    demo_row = {
        "claim_id": "C099",
        "organization": "Fraudulent Carbon Ops",
        "claimed_reduction_tco2e": 3200.0,
        "historical_avg_emissions_tco2e": 10000.0,
        "current_emissions_tco2e": 9800.0,
        "historical_avg_energy_mwh": 5000.0,
        "current_energy_mwh": 6100.0,
    }
    out = fraud_df.loc[fraud_df["claim_id"] != "C099"].copy()
    out = pd.concat([out, pd.DataFrame([demo_row])], ignore_index=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic carbon credit claim datasets.")
    parser.add_argument("--rows", type=int, default=500, help="Number of rows in fraud dataset.")
    parser.add_argument("--fraud-ratio", type=float, default=0.08, help="Fraction of fraudulent claims.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--normal-out", default="data/large_claims.csv", help="Output path for normal-only dataset.")
    parser.add_argument(
        "--fraud-out",
        default="data/large_claims_fraud.csv",
        help="Output path for dataset containing both normal and fraud claims.",
    )
    parser.add_argument(
        "--include-c099",
        action="store_true",
        help="Append fixed demo fraud claim C099 to fraud output file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.rows < 50:
        raise ValueError("rows must be at least 50 for stable anomaly scoring context.")
    if not (0 < args.fraud_ratio < 0.5):
        raise ValueError("fraud-ratio must be between 0 and 0.5.")

    normal_df, fraud_df = generate_dataset(
        rows=args.rows,
        fraud_ratio=args.fraud_ratio,
        seed=args.seed,
    )
    if args.include_c099:
        fraud_df = append_c099_demo_claim(fraud_df)

    normal_path = Path(args.normal_out)
    fraud_path = Path(args.fraud_out)
    normal_path.parent.mkdir(parents=True, exist_ok=True)
    fraud_path.parent.mkdir(parents=True, exist_ok=True)

    normal_df.to_csv(normal_path, index=False)
    fraud_df.to_csv(fraud_path, index=False)

    print(f"Saved normal dataset: {normal_path} ({len(normal_df)} rows)")
    print(f"Saved fraud dataset: {fraud_path} ({len(fraud_df)} rows)")
