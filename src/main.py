from __future__ import annotations

import argparse
from pathlib import Path

from data_loader import get_dataset_options, load_claims
from features import build_features, model_feature_matrix
from model import detect_anomalies
from scoring import assign_scores


def run_pipeline(
    input_csv: str | None,
    output_csv: str,
    contamination: float,
    dataset: str | None = None,
) -> None:
    claims_df = load_claims(path=input_csv, dataset=dataset)
    feature_df = build_features(claims_df)
    X = model_feature_matrix(feature_df)
    anomaly_df = detect_anomalies(X, contamination=contamination)

    result = feature_df.join(anomaly_df)
    result = assign_scores(result)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    total = len(result)
    high_risk = int((result["risk_level"] == "HIGH").sum())
    medium_risk = int((result["risk_level"] == "MEDIUM").sum())
    low_risk = int((result["risk_level"] == "LOW").sum())

    print(f"Processed {total} claims.")
    print(f"LOW: {low_risk}, MEDIUM: {medium_risk}, HIGH: {high_risk}")
    print(f"Saved results to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-driven carbon credit fraud detection.")
    parser.add_argument("--input", help="Path to input claims CSV file.")
    parser.add_argument(
        "--dataset",
        choices=get_dataset_options(),
        help="Built-in dataset option from data_loader.",
    )
    parser.add_argument("--output", default="outputs/results.csv", help="Path to output CSV file.")
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.2,
        help="Estimated fraction of anomalies in data (0 < value < 0.5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input, args.output, args.contamination, dataset=args.dataset)
