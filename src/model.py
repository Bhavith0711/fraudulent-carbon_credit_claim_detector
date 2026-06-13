from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(X: pd.DataFrame, contamination: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    # decision_function: larger values are more normal; invert so larger means more anomalous
    anomaly_score = -model.decision_function(X)
    model_pred = model.predict(X)  # -1 anomaly, 1 normal
    model_flag = (model_pred == -1).astype(int)

    return pd.DataFrame(
        {
            "anomaly_score": anomaly_score,
            "model_flag": model_flag,
        }
    )
