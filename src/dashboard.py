from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from .features import build_features, model_feature_matrix
    from .model import detect_anomalies
    from .scoring import assign_scores
except ImportError:
    from features import build_features, model_feature_matrix
    from model import detect_anomalies
    from scoring import assign_scores


def run_scoring(df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    features = build_features(df)
    anomaly_df = detect_anomalies(model_feature_matrix(features), contamination=contamination)
    return assign_scores(features.join(anomaly_df))


def main() -> None:
    st.set_page_config(page_title="Carbon Fraud Detection", layout="wide")
    st.title("AI-Driven Carbon Credit Fraud Detection")
    st.caption("Preview dashboard for credibility scoring, risk classification, and action recommendations.")

    contamination = st.sidebar.slider("Anomaly contamination", 0.05, 0.40, 0.20, 0.01)
    use_fraud_demo = st.sidebar.checkbox("Use fraud demo dataset", value=True)

    default_path = "data/sample_claims_fraud.csv" if use_fraud_demo else "data/sample_claims.csv"
    csv_path = st.sidebar.text_input("CSV path", value=default_path)

    try:
        df = pd.read_csv(Path(csv_path))
    except Exception as exc:
        st.error(f"Failed to load CSV: {exc}")
        return

    result = run_scoring(df, contamination=contamination)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Claims", len(result))
    c2.metric("High Risk", int((result["risk_level"] == "HIGH").sum()))
    c3.metric("Average Credibility", f"{result['credibility_score'].mean():.2f}")

    st.subheader("Ranked Claims")
    view_cols = [
        "claim_id",
        "organization",
        "claimed_reduction_tco2e",
        "expected_reduction_tco2e",
        "reduction_gap_tco2e",
        "energy_change_pct",
        "anomaly_score",
        "credibility_score",
        "risk_level",
        "recommendation",
    ]
    st.dataframe(result.sort_values("credibility_score", ascending=True)[view_cols], width="stretch")

    st.subheader("Fraud Example Spotlight")
    claim_id = st.selectbox("Claim to inspect", options=result["claim_id"].tolist(), index=0)
    row = result.loc[result["claim_id"] == claim_id].iloc[0]

    st.json(
        {
            "claim_id": row["claim_id"],
            "organization": row["organization"],
            "credibility_score": float(row["credibility_score"]),
            "risk_level": row["risk_level"],
            "recommendation": row["recommendation"],
            "flag_reasons": [
                "Large reduction gap"
                if abs(row["reduction_gap_tco2e"]) > max(100.0, 0.25 * row["claimed_reduction_tco2e"])
                else "Reduction gap within expected range",
                "Energy trend inconsistent with claimed savings"
                if row["energy_change_pct"] > 8
                and row["claimed_reduction_tco2e"] > row["expected_reduction_tco2e"]
                else "Energy trend appears consistent",
                "Model anomaly detected" if int(row["model_flag"]) == 1 else "Model did not flag anomaly",
            ],
        }
    )


if __name__ == "__main__":
    main()
