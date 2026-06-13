from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

try:
    from .features import build_features, model_feature_matrix
    from .model import detect_anomalies
    from .scoring import assign_scores
except ImportError:
    from features import build_features, model_feature_matrix
    from model import detect_anomalies
    from scoring import assign_scores


class ClaimRecord(BaseModel):
    claim_id: str
    organization: str
    claimed_reduction_tco2e: float = Field(ge=0)
    historical_avg_emissions_tco2e: float = Field(gt=0)
    current_emissions_tco2e: float = Field(ge=0)
    historical_avg_energy_mwh: float = Field(gt=0)
    current_energy_mwh: float = Field(ge=0)


class ScoreRequest(BaseModel):
    claims: List[ClaimRecord]
    contamination: float = Field(default=0.2, gt=0, lt=0.5)


MODEL_NAME = "IsolationForest"
MODEL_VERSION = "1.0.0"
API_VERSION = "1.1.0"
MIN_BATCH_SIZE = 5
MAX_BATCH_SIZE = 500
RATE_LIMIT_REQUESTS_PER_MIN = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MIN", "60"))
SERVICE_START_TS = time.time()
_rate_limit_state: Dict[str, List[float]] = {}

logger = logging.getLogger("carbon_fraud_api")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _run_scoring(df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    feature_df = build_features(df)
    model_df = detect_anomalies(model_feature_matrix(feature_df), contamination=contamination)
    result = assign_scores(feature_df.join(model_df))
    return result


def _flags(row: pd.Series) -> List[str]:
    flags: List[str] = []
    if abs(row["reduction_gap_tco2e"]) > max(100.0, 0.25 * row["claimed_reduction_tco2e"]):
        flags.append("Claim deviates materially from expected reduction.")
    if row["energy_change_pct"] > 8 and row["claimed_reduction_tco2e"] > row["expected_reduction_tco2e"]:
        flags.append("Energy usage rose while large reduction was claimed.")
    if int(row["model_flag"]) == 1:
        flags.append("IsolationForest flagged claim as anomalous.")
    if not flags:
        flags.append("No major anomaly flags.")
    return flags


def _to_output_row(row: pd.Series) -> Dict[str, Any]:
    return {
        "claim_id": row["claim_id"],
        "organization": row["organization"],
        "expected_reduction_tco2e": round(float(row["expected_reduction_tco2e"]), 2),
        "reduction_gap_tco2e": round(float(row["reduction_gap_tco2e"]), 2),
        "energy_change_pct": round(float(row["energy_change_pct"]), 2),
        "anomaly_score": round(float(row["anomaly_score"]), 4),
        "model_flag": int(row["model_flag"]),
        "credibility_score": round(float(row["credibility_score"]), 2),
        "risk_level": row["risk_level"],
        "recommendation": row["recommendation"],
        "flags": _flags(row),
    }


def _validate_api_key(x_api_key: str | None) -> None:
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        return
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API key.")


def _enforce_rate_limit(client_id: str) -> None:
    now = time.time()
    window_start = now - 60
    history = [ts for ts in _rate_limit_state.get(client_id, []) if ts >= window_start]
    if len(history) >= RATE_LIMIT_REQUESTS_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    history.append(now)
    _rate_limit_state[client_id] = history


def _readiness_check() -> Dict[str, Any]:
    sample_path = Path("data/sample_claims_fraud.csv")
    if not sample_path.exists():
        return {"ready": False, "reason": "missing data/sample_claims_fraud.csv"}
    try:
        sample_df = pd.read_csv(sample_path)
        if sample_df.empty:
            return {"ready": False, "reason": "sample dataset is empty"}
    except Exception as exc:
        return {"ready": False, "reason": f"failed to load sample dataset: {exc}"}
    return {"ready": True}


def _model_meta() -> Dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "api_version": API_VERSION,
    }


app = FastAPI(title="Carbon Credit Fraud Detection API", version=API_VERSION)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "api_version": API_VERSION}


@app.get("/ready")
def ready() -> Dict[str, Any]:
    readiness = _readiness_check()
    status = "ready" if readiness["ready"] else "not_ready"
    return {
        "status": status,
        "uptime_seconds": round(time.time() - SERVICE_START_TS, 2),
        **readiness,
        "metadata": _model_meta(),
    }


@app.get("/preview/fraud-example")
def fraud_example(
    request: Request,
    contamination: float = 0.2,
    x_api_key: str | None = Header(default=None),
) -> Dict[str, Any]:
    _validate_api_key(x_api_key)
    client_id = request.client.host if request.client else "unknown"
    _enforce_rate_limit(client_id)

    sample_path = Path("data/sample_claims_fraud.csv")
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="data/sample_claims_fraud.csv not found.")

    claims_df = pd.read_csv(sample_path)
    result = _run_scoring(claims_df, contamination=contamination)
    fraud_row = result.loc[result["claim_id"] == "C099"]
    if fraud_row.empty:
        raise HTTPException(status_code=404, detail="Fraud demo claim C099 not present.")
    row = fraud_row.iloc[0]
    logger.info(
        "fraud_preview_scored claims=%s contamination=%.3f client=%s",
        len(claims_df),
        contamination,
        client_id,
    )
    return {"metadata": _model_meta(), "demo_claim": _to_output_row(row)}


@app.post("/score-claims")
def score_claims(
    payload: ScoreRequest,
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> Dict[str, Any]:
    _validate_api_key(x_api_key)
    client_id = request.client.host if request.client else "unknown"
    _enforce_rate_limit(client_id)

    total_claims = len(payload.claims)
    if total_claims < MIN_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Provide at least {MIN_BATCH_SIZE} claims so the anomaly model has enough context.",
        )
    if total_claims > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large. Maximum supported claims per request is {MAX_BATCH_SIZE}.",
        )

    claims_df = pd.DataFrame([item.model_dump() for item in payload.claims])
    result = _run_scoring(claims_df, contamination=payload.contamination)

    out = [_to_output_row(row) for _, row in result.iterrows()]
    logger.info(
        "batch_scored claims=%s contamination=%.3f high_risk=%s client=%s",
        total_claims,
        payload.contamination,
        sum(1 for row in out if row["risk_level"] == "HIGH"),
        client_id,
    )
    return {
        "total_claims": len(out),
        "high_risk_count": sum(1 for row in out if row["risk_level"] == "HIGH"),
        "metadata": _model_meta(),
        "results": out,
    }
