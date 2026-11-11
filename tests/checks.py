#!/usr/bin/env python3
"""Task-specific verifier checks for the fairness classification task."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

APP_DIR = Path(os.environ.get("OTTER_APP_DIR", "/app"))


def app_path(*parts: str) -> Path:
    """Resolve a path relative to the working directory."""
    return APP_DIR.joinpath(*parts)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file or raise a helpful explanation."""
    if not path.exists():
        raise AssertionError(f"Missing JSON file: {path}")
    try:
        contents = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Invalid JSON in {path}: line {exc.lineno}, col {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(contents, dict):
        raise TypeError(f"Expected JSON object in {path}; found {type(contents).__name__}")
    return contents


def require_keys(obj: dict[str, Any], keys: set[str], label: str) -> None:
    """Assert that a dict contains the required keys."""
    if not isinstance(obj, dict):
        raise TypeError(f"{label} must be a dict; got {type(obj).__name__}")
    missing = sorted(keys - set(obj))
    if missing:
        raise AssertionError(f"{label} missing required keys: {missing}")


def check_data_deliverables() -> str:
    """Validate the data-preparation artifacts exist."""
    for path in [
        app_path("artifacts/dataset_manifest.json"),
        app_path("artifacts/processed_data.npz"),
    ]:
        assert path.exists(), f"Missing: {path}"
    return "data deliverables exist"


def check_data_schema() -> str:
    """Validate the data manifest structure."""
    manifest = load_json(app_path("artifacts/dataset_manifest.json"))
    require_keys(
        manifest,
        {"source", "n_samples", "n_features", "target_column", "protected_column", "splits"},
        "dataset_manifest.json",
    )
    assert manifest["n_samples"] > 0, "n_samples must be positive"
    assert manifest["n_features"] > 0, "n_features must be positive"
    assert sum(manifest["splits"].values()) == manifest["n_samples"], (
        "split counts must sum to n_samples"
    )
    return "data schema valid"


def check_model_deliverables() -> str:
    """Validate the model artifacts exist."""
    for path in [
        app_path("artifacts/metrics.json"),
        app_path("artifacts/predictions.csv"),
        app_path("models/trained_model.pkl"),
    ]:
        assert path.exists(), f"Missing: {path}"
    return "model deliverables exist"


def check_metrics_schema() -> str:
    """Validate the metric payload structure and value ranges."""
    metrics = load_json(app_path("artifacts/metrics.json"))
    require_keys(metrics, {"auroc", "max_platform_fpr_disparity"}, "metrics.json")
    assert 0.0 <= metrics["auroc"] <= 1.0, "auroc must be in [0, 1]"
    assert 0.0 <= metrics["max_platform_fpr_disparity"] <= 1.0, (
        "max_platform_fpr_disparity must be in [0, 1]"
    )
    return "metrics schema valid"


def check_metrics_thresholds() -> str:
    """Set a safety floor below the human threshold but above random performance."""
    metrics = load_json(app_path("artifacts/metrics.json"))
    assert metrics["auroc"] >= 0.55, f"AUROC below sanity floor: {metrics['auroc']}"
    assert metrics["max_platform_fpr_disparity"] <= 0.75, (
        "FPR disparity should remain reasonably bounded"
    )
    return "metrics above baseline"


def check_final_deliverables() -> str:
    """Check that the final reporting files exist."""
    for path in [
        app_path("artifacts/final_results.json"),
        app_path("reports/report.md"),
    ]:
        assert path.exists(), f"Missing: {path}"
    return "final deliverables exist"


def check_report_content() -> str:
    """Check the report includes the required sections and detail."""
    report = app_path("reports/report.md").read_text().lower()
    required_terms = ["model", "fairness", "platform", "auroc", "results"]
    missing = [term for term in required_terms if term not in report]
    assert not missing, f"Report missing required terms: {missing}"
    word_count = len(report.split())
    assert word_count >= 40, f"Report too short: {word_count} words"
    return "report content valid"


def _criterion(id: str, fn, milestone_id: str = "final") -> dict[str, Any]:
    """Run a check and return the criterion structure expected by the harness."""
    try:
        detail = fn()
        return {
            "id": id,
            "passed": True,
            "detail": detail or "passed",
            "milestone_id": milestone_id,
        }
    except Exception:  # noqa: BLE001
        return {
            "id": id,
            "passed": False,
            "detail": traceback.format_exc(limit=6),
            "milestone_id": milestone_id,
        }


def evaluate(context: dict) -> dict:
    """Return the check results and metric payload expected by the Otter harness."""
    metrics_path = app_path("artifacts/metrics.json")
    metrics = load_json(metrics_path) if metrics_path.exists() else {}

    criteria = [
        _criterion("data_deliverables", check_data_deliverables, "data-preparation"),
        _criterion("data_schema", check_data_schema, "data-preparation"),
        _criterion("model_deliverables", check_model_deliverables, "model-evaluation"),
        _criterion("metrics_schema", check_metrics_schema, "model-evaluation"),
        _criterion("metrics_thresholds", check_metrics_thresholds, "model-evaluation"),
        _criterion("final_deliverables", check_final_deliverables, "deliverables-and-report"),
        _criterion("report_content", check_report_content, "deliverables-and-report"),
    ]

    return {
        "criteria": criteria,
        "metrics": {
            "auroc": float(metrics.get("auroc", 0.0)),
            "max_platform_fpr_disparity": float(metrics.get("max_platform_fpr_disparity", 1.0)),
        },
    }