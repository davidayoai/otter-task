#!/usr/bin/env python3
"""Reference pipeline for [YOUR TASK NAME].

This module contains the oracle solution. solve.sh copies it into /app/src/
and run.py calls run_full() to produce all required artifacts.

The agent is expected to produce code that generates the same artifacts
(as defined in instruction.md). checks.py validates those artifacts.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(os.environ.get("OTTER_APP_DIR", "/app"))
ARTIFACTS = APP_DIR / "artifacts"
REPORTS = APP_DIR / "reports"
MODELS = APP_DIR / "models"


def _default_synthetic_dataset(path: Path) -> pd.DataFrame:
    """Generate a small synthetic dataset if the task input is absent."""
    rng = np.random.default_rng(42)
    n_rows = 1200
    platform = rng.integers(0, 2, size=n_rows)
    age = rng.normal(45, 12, size=n_rows)
    income = rng.normal(60, 20, size=n_rows)
    credit = rng.normal(680, 60, size=n_rows)
    prior_default = rng.integers(0, 3, size=n_rows)
    risk_score = (
        0.9 * (age - 45) / 12
        + 0.7 * (income - 60) / 20
        + 0.8 * (credit - 680) / 60
        + 0.25 * prior_default
    )
    risk_score += 0.1 * platform
    probability = 1.0 / (1.0 + np.exp(-risk_score))
    target = rng.random(n_rows) < probability

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "credit_score": credit,
            "prior_default_count": prior_default,
            "platform": platform.astype(str),
            "target": target.astype(int),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _coerce_target(data: pd.DataFrame, candidate_columns: list[str]) -> tuple[str, pd.Series]:
    for column in candidate_columns:
        if column in data.columns:
            series = data[column]
            unique_values = sorted({value for value in series.dropna().tolist()})
            if len(unique_values) == 2:
                value_to_int = {value: int(idx) for idx, value in enumerate(unique_values)}
                return column, series.map(value_to_int)
            if len(unique_values) > 2:
                ordered = sorted(unique_values)
                value_to_int = {value: int(value == ordered[-1]) for value in ordered}
                return column, series.map(value_to_int)
    if "target" not in data.columns:
        data["target"] = (data.iloc[:, -1].astype(float) > data.iloc[:, -1].median()).astype(int)
    target_series = data["target"].astype(int)
    return "target", target_series


def _resolve_protected_column(data: pd.DataFrame) -> str:
    preferred = [
        "platform",
        "group",
        "protected",
        "protected_group",
        "gender",
        "sex",
        "race",
        "device",
        "site",
        "region",
        "segment",
    ]
    for name in preferred:
        if name in data.columns:
            return name
    for column in data.select_dtypes(include=["object", "category"]).columns:
        if column not in {"target"}:
            return column
    for column in data.columns:
        if column not in {"target"}:
            return column
    return "platform"


def _resolve_data_path(path: str | Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    for env_name in ("OTTER_DATA_PATH", "DATA_PATH"):
        value = os.environ.get(env_name)
        if value:
            candidate = Path(value)
            if candidate.exists():
                return candidate
    candidates = [
        APP_DIR / "data" / "data.csv",
        APP_DIR / "data" / "input_data.csv",
        APP_DIR / "data" / "input_data.npz",
        Path(__file__).resolve().parents[2] / "environment" / "task_inputs" / "data.csv",
        Path(__file__).resolve().parents[2] / "environment" / "task_inputs" / "input_data.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    fallback = APP_DIR / "data" / "data.csv"
    _default_synthetic_dataset(fallback)
    return fallback


def _load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".npz":
        arrays = np.load(path)
        if "data" in arrays:
            data = arrays["data"]
            if isinstance(data, np.ndarray):
                return pd.DataFrame(data)
        if "X" in arrays and "y" in arrays:
            frame = pd.DataFrame(arrays["X"])
            if "feature_names" in arrays:
                frame.columns = arrays["feature_names"].tolist()
            frame["target"] = arrays["y"]
            return frame
        raise ValueError(f"Unsupported .npz structure in {path}")
    return pd.read_csv(path)


# -------------------------------------------------------------------------
# Milestone 1: Data loading and preprocessing
# -------------------------------------------------------------------------

def load_and_preprocess(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Load data, infer target/protected columns, and save a manifest."""
    source_path = _resolve_data_path(data_path)
    df = _load_dataframe(source_path)
    if df.empty:
        raise ValueError(f"Input dataset {source_path} is empty")

    target_column, target_series = _coerce_target(df, ["target", "label", "outcome", "y", "class"])
    protected_column = _resolve_protected_column(df)
    if protected_column == target_column:
        protected_column = "platform" if "platform" in df.columns else next(
            col for col in df.columns if col not in {target_column}
        )
    if protected_column not in df.columns:
        df[protected_column] = np.where(df.iloc[:, 0].astype(float) > df.iloc[:, 0].median(), "A", "B")
    df = df.copy()
    df[target_column] = target_series.astype(int)

    feature_frame = df.drop(columns=[target_column, protected_column], errors="ignore")
    if feature_frame.empty:
        raise ValueError("No feature columns available after dropping target and protected fields")

    output = Path(output_dir) if output_dir else ARTIFACTS
    output.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, _protected_train, protected_test = train_test_split(
        feature_frame,
        df[target_column],
        df[protected_column],
        test_size=0.25,
        stratify=df[target_column],
        random_state=seed,
    )
    splits = {"train": len(X_train), "test": len(X_test)}

    np.savez_compressed(
        output / "processed_data.npz",
        X_train=X_train.to_numpy(),
        X_test=X_test.to_numpy(),
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
        protected_train=_protected_train.to_numpy(),
        protected_test=protected_test.to_numpy(),
    )

    manifest = {
        "source": str(source_path),
        "n_samples": len(df),
        "n_features": len(feature_frame.columns),
        "target_column": target_column,
        "protected_column": protected_column,
        "splits": splits,
        "feature_names": feature_frame.columns.tolist(),
        "preprocessing_steps": ["imputation", "one_hot_encoding", "standard_scaling"],
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "n_samples": len(df),
        "n_features": len(feature_frame.columns),
        "target_column": target_column,
        "protected_column": protected_column,
        "splits": splits,
    }


# -------------------------------------------------------------------------
# Milestone 2: Model training and evaluation
# -------------------------------------------------------------------------

def _build_model_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_columns = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_and_evaluate(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a classification model and save fairness metrics."""
    source_path = _resolve_data_path(data_path)
    df = _load_dataframe(source_path)
    target_column, target_series = _coerce_target(df, ["target", "label", "outcome", "y", "class"])
    protected_column = _resolve_protected_column(df)
    if protected_column == target_column:
        protected_column = "platform" if "platform" in df.columns else next(
            col for col in df.columns if col not in {target_column}
        )
    if protected_column not in df.columns:
        df[protected_column] = np.where(df.iloc[:, 0].astype(float) > df.iloc[:, 0].median(), "A", "B")
    df = df.copy()
    df[target_column] = target_series.astype(int)

    feature_frame = df.drop(columns=[target_column, protected_column], errors="ignore")
    if feature_frame.empty:
        raise ValueError("No feature columns available after dropping target and protected variables")

    X_train, X_test, y_train, y_test, _protected_train, protected_test = train_test_split(
        feature_frame,
        df[target_column],
        df[protected_column],
        test_size=0.25,
        stratify=df[target_column],
        random_state=seed,
    )

    model = _build_model_pipeline(X_train)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    auroc = float(roc_auc_score(y_test, probabilities))
    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions))

    fpr_by_group: dict[str, float] = {}
    for group in sorted(protected_test.unique()):
        mask = protected_test.astype(str) == str(group)
        negatives = mask & (y_test == 0)
        if not negatives.any():
            fpr_by_group[str(group)] = 0.0
            continue
        false_positives = np.sum((predictions[mask] == 1) & (y_test[mask] == 0))
        fpr_by_group[str(group)] = float(false_positives / negatives.sum())
    max_platform_fpr_disparity = float(max(fpr_by_group.values()) - min(fpr_by_group.values())) if fpr_by_group else 0.0

    out_dir = Path(output_dir) if output_dir else ARTIFACTS
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    metrics = {
        "auroc": round(auroc, 6),
        "accuracy": round(accuracy, 6),
        "f1": round(f1, 6),
        "max_platform_fpr_disparity": round(max_platform_fpr_disparity, 6),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    preds = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "prediction": predictions,
            "probability": probabilities,
            protected_column: protected_test.astype(str).to_numpy(),
        }
    )
    preds.to_csv(out_dir / "predictions.csv", index=False)

    with (MODELS / "trained_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    final_results = {
        "model": "logistic_regression",
        "target_column": target_column,
        "protected_column": protected_column,
        "metrics": metrics,
    }
    (out_dir / "final_results.json").write_text(json.dumps(final_results, indent=2))

    report_lines = [
        "# Fairness Audit Report",
        "",
        f"- Target column: {target_column}",
        f"- Protected column: {protected_column}",
        f"- Test AUROC: {metrics['auroc']:.4f}",
        f"- Max platform FPR disparity: {metrics['max_platform_fpr_disparity']:.4f}",
        "",
        "The model uses a standard logistic-regression pipeline with median imputation, one-hot encoding for categorical features, and standard scaling for numeric features.",
        "This setup keeps the model interpretable while providing stable probability estimates for a fairness audit.",
    ]
    (REPORTS / "report.md").write_text("\n".join(report_lines) + "\n")

    return metrics


# -------------------------------------------------------------------------
# Full end-to-end pipeline
# -------------------------------------------------------------------------

def run_full(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the data-prep and model-evaluation lifecycle end-to-end."""
    out_dir = Path(output_dir) if output_dir else ARTIFACTS
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    load_and_preprocess(data_path, out_dir, seed)
    return train_and_evaluate(data_path, out_dir, seed)


if __name__ == "__main__":
    run_full()
