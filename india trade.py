#!/usr/bin/env python3
"""
Trade India - Single-file utility

This script provides a clean, minimal, and runnable command-line tool for:
- Loading a CSV of India trade data (imports/exports or any numeric dataset)
- Cleaning and summarizing the data
- Training a lightweight model (RandomForestRegressor) to predict a target column
- Providing simple anomaly detection using IsolationForest

All in one file with zero external project dependencies beyond common Python libs.

Usage examples:
  python3 "india trade.py" summary --file data.csv
  python3 "india trade.py" train --file data.csv --target Export_Value
  python3 "india trade.py" predict --model model.joblib --row '{"Feature1": 1.2, "Feature2": 3}'
  python3 "india trade.py" anomalies --file data.csv

If you do not have scikit-learn, install it via:
  pip install scikit-learn pandas numpy joblib
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV has no rows.")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicate rows
    df = df.drop_duplicates()
    # Standardize column names (strip and replace spaces with underscores)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def summarize_dataframe(df: pd.DataFrame) -> str:
    buf = []
    buf.append("Columns: " + ", ".join(df.columns))
    buf.append("")
    buf.append("Shape: " + str(df.shape))
    buf.append("")
    # Basic numeric summary
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        desc = num_df.describe().to_string()
        buf.append("Numeric summary:\n" + desc)
    else:
        buf.append("No numeric columns detected.")
    return "\n".join(buf)


def split_features_target(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not in data.")
    y = df[target]
    X = df.drop(columns=[target])
    # Keep only numeric features for this simple baseline model
    X = X.select_dtypes(include=[np.number]).copy()
    if X.empty:
        raise ValueError("No numeric features available after dropping the target.")
    # Drop rows with missing values in features or target
    valid = X.notnull().all(axis=1) & y.notnull()
    X = X.loc[valid]
    y = y.loc[valid]
    return X, y


def train_model(
    df: pd.DataFrame, target: str, model_out: Path, test_size: float = 0.2, random_state: int = 42
) -> None:
    X, y = split_features_target(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    joblib.dump({"model": model, "features": list(X.columns), "target": target}, model_out)
    print(json.dumps({"status": "ok", "r2": r2, "mae": mae, "model_path": str(model_out)}, indent=2))


def load_model_bundle(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_row(model_path: Path, row_json: str) -> float:
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    features: List[str] = bundle["features"]
    data = json.loads(row_json)
    # Build row in the learned feature order; missing features become NaN
    values = [data.get(feat, np.nan) for feat in features]
    arr = np.array(values, dtype=float).reshape(1, -1)
    if np.isnan(arr).any():
        missing = [features[i] for i, v in enumerate(values) if v is None or (isinstance(v, float) and np.isnan(v))]
        raise ValueError(f"Missing feature values for: {missing}")
    pred = float(model.predict(arr)[0])
    return pred


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> pd.Series:
    num = df.select_dtypes(include=[np.number]).dropna()
    if num.empty:
        raise ValueError("No numeric columns available for anomaly detection.")
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    labels = iso.fit_predict(num.values)
    # -1 indicates anomaly
    anomalies = pd.Series((labels == -1), index=num.index)
    return anomalies


def cmd_summary(args: argparse.Namespace) -> None:
    df = clean_dataframe(load_csv(Path(args.file)))
    print(summarize_dataframe(df))


def cmd_train(args: argparse.Namespace) -> None:
    df = clean_dataframe(load_csv(Path(args.file)))
    model_out = Path(args.model)
    train_model(df, args.target, model_out)


def cmd_predict(args: argparse.Namespace) -> None:
    pred = predict_row(Path(args.model), args.row)
    print(json.dumps({"prediction": pred}, indent=2))


def cmd_anomalies(args: argparse.Namespace) -> None:
    df = clean_dataframe(load_csv(Path(args.file)))
    anomalies = detect_anomalies(df, contamination=args.contamination)
    out = {
        "total_numeric_rows": int(anomalies.shape[0]),
        "anomaly_count": int(anomalies.sum()),
        "anomaly_indices": [int(i) for i in anomalies[anomalies].index[:100]],  # cap list size
    }
    print(json.dumps(out, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trade India - single-file analytics utility")
    sub = p.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summary", help="Print data summary")
    p_sum.add_argument("--file", required=True, help="Path to CSV file")
    p_sum.set_defaults(func=cmd_summary)

    p_train = sub.add_parser("train", help="Train a RandomForest model on target column")
    p_train.add_argument("--file", required=True, help="Path to CSV file")
    p_train.add_argument("--target", required=True, help="Target column to predict")
    p_train.add_argument("--model", required=False, default="model.joblib", help="Output model path")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Predict using a saved model and a single row JSON")
    p_pred.add_argument("--model", required=True, help="Path to model.joblib")
    p_pred.add_argument("--row", required=True, help='JSON string for a single row of features')
    p_pred.set_defaults(func=cmd_predict)

    p_an = sub.add_parser("anomalies", help="Detect anomalies using IsolationForest")
    p_an.add_argument("--file", required=True, help="Path to CSV file")
    p_an.add_argument("--contamination", type=float, default=0.05, help="Proportion of anomalies")
    p_an.set_defaults(func=cmd_anomalies)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        args.func(args)
        return 0
    except Exception as exc:  # surface concise error
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

