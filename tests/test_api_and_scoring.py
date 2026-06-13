from __future__ import annotations

import os

import pandas as pd
from fastapi.testclient import TestClient

from src import api as api_module
from src.api import app
from src.features import build_features, model_feature_matrix
from src.model import detect_anomalies
from src.scoring import assign_scores


def _sample_payload_from_csv(path: str) -> dict:
    df = pd.read_csv(path)
    claims = df.to_dict(orient="records")
    return {"claims": claims, "contamination": 0.2}


def _client() -> TestClient:
    return TestClient(app)


def setup_function() -> None:
    api_module._rate_limit_state.clear()
    os.environ.pop("API_KEY", None)


def test_health_and_ready_endpoints() -> None:
    client = _client()
    health = client.get("/health")
    ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "api_version" in health.json()

    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] in {"ready", "not_ready"}
    assert "metadata" in body
    assert body["metadata"]["model_name"] == "IsolationForest"


def test_fraud_preview_returns_demo_claim_with_flags() -> None:
    client = _client()
    response = client.get("/preview/fraud-example")
    assert response.status_code == 200

    body = response.json()
    assert "metadata" in body
    assert "demo_claim" in body
    assert body["demo_claim"]["claim_id"] == "C099"
    assert isinstance(body["demo_claim"]["flags"], list)
    assert len(body["demo_claim"]["flags"]) >= 1


def test_score_claims_rejects_small_batch() -> None:
    client = _client()
    payload = _sample_payload_from_csv("data/sample_claims.csv")
    payload["claims"] = payload["claims"][:4]

    response = client.post("/score-claims", json=payload)
    assert response.status_code == 400
    assert "at least" in response.json()["detail"]


def test_score_claims_returns_metadata_and_results() -> None:
    client = _client()
    payload = _sample_payload_from_csv("data/sample_claims_fraud.csv")

    response = client.post("/score-claims", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["total_claims"] == len(payload["claims"])
    assert "metadata" in body
    assert "results" in body
    assert any(item["claim_id"] == "C099" for item in body["results"])


def test_auth_required_when_api_key_configured() -> None:
    os.environ["API_KEY"] = "secret-key"
    client = _client()
    payload = _sample_payload_from_csv("data/sample_claims_fraud.csv")

    no_key = client.post("/score-claims", json=payload)
    bad_key = client.post("/score-claims", json=payload, headers={"x-api-key": "wrong"})
    good_key = client.post("/score-claims", json=payload, headers={"x-api-key": "secret-key"})

    assert no_key.status_code == 401
    assert bad_key.status_code == 401
    assert good_key.status_code == 200


def test_rate_limit_enforced() -> None:
    api_module.RATE_LIMIT_REQUESTS_PER_MIN = 2
    client = _client()

    r1 = client.get("/preview/fraud-example")
    r2 = client.get("/preview/fraud-example")
    r3 = client.get("/preview/fraud-example")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429


def test_pipeline_flags_demo_claim_as_high_risk() -> None:
    df = pd.read_csv("data/sample_claims_fraud.csv")
    features = build_features(df)
    anomaly_df = detect_anomalies(model_feature_matrix(features), contamination=0.2)
    scored = assign_scores(features.join(anomaly_df))

    row = scored.loc[scored["claim_id"] == "C099"].iloc[0]
    assert row["risk_level"] == "HIGH"
    assert float(row["credibility_score"]) <= 20.0


def test_scoring_output_bounds_and_labels() -> None:
    df = pd.read_csv("data/sample_claims.csv")
    features = build_features(df)
    anomaly_df = detect_anomalies(model_feature_matrix(features), contamination=0.2)
    scored = assign_scores(features.join(anomaly_df))

    assert scored["credibility_score"].between(0, 100).all()
    assert set(scored["risk_level"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})
    assert scored["recommendation"].notna().all()
