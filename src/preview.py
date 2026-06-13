from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

try:
    from .api import _run_scoring, _to_output_row
except ImportError:
    from api import _run_scoring, _to_output_row


def run_fraud_preview(data_path: str, contamination: float) -> dict:
    df = pd.read_csv(Path(data_path))
    result = _run_scoring(df, contamination=contamination)
    fraud_row = result.loc[result["claim_id"] == "C099"]
    if fraud_row.empty:
        raise ValueError("Fraud demo claim C099 not found in input dataset.")
    return _to_output_row(fraud_row.iloc[0])


def launch_services(api_port: int, dashboard_port: int) -> None:
    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(api_port),
    ]
    dashboard_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/dashboard.py",
        "--server.port",
        str(dashboard_port),
        "--server.headless",
        "true",
    ]

    print("\nLaunching services...")
    api_proc = subprocess.Popen(api_cmd)
    dashboard_proc = subprocess.Popen(dashboard_cmd)
    print(f"- API docs: http://127.0.0.1:{api_port}/docs")
    print(f"- API fraud preview: http://127.0.0.1:{api_port}/preview/fraud-example")
    print(f"- Dashboard: http://127.0.0.1:{dashboard_port}")
    print("\nPress Ctrl+C to stop both services.")

    try:
        while True:
            if api_proc.poll() is not None:
                raise RuntimeError("API server stopped unexpectedly.")
            if dashboard_proc.poll() is not None:
                raise RuntimeError("Dashboard server stopped unexpectedly.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        for proc in (api_proc, dashboard_proc):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command local preview for carbon credit fraud detection."
    )
    parser.add_argument(
        "--input",
        default="data/sample_claims_fraud.csv",
        help="Path to CSV used for preview scoring.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.2,
        help="Estimated anomaly fraction for IsolationForest (0 < value < 0.5).",
    )
    parser.add_argument(
        "--start-services",
        action="store_true",
        help="Start FastAPI and Streamlit after printing fraud preview output.",
    )
    parser.add_argument("--api-port", type=int, default=8000, help="FastAPI port.")
    parser.add_argument("--dashboard-port", type=int, default=8501, help="Streamlit port.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preview = run_fraud_preview(args.input, args.contamination)
    print("=== Fraud Preview (C099) ===")
    for key, value in preview.items():
        print(f"{key}: {value}")

    if args.start_services:
        launch_services(api_port=args.api_port, dashboard_port=args.dashboard_port)
