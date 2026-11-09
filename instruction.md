Train a tabular classification model on the provided fairness dataset and optimize for both predictive quality and platform-level false-positive parity.

Requirements:
- Use the data available from `/app/data/data.csv`, `/app/data/input_data.csv`, or `/app/data/input_data.npz` if present.
- If a protected attribute named `platform` is present, use it to audit platform false-positive rate parity; otherwise infer a comparable categorical group column from the dataset.
- Train and evaluate a model that achieves strong discrimination while keeping the worst-case platform FPR disparity small.
- Write your final evaluation metrics to `/app/artifacts/metrics.json` as a JSON object with exactly these keys:
  {
    "auroc": <float between 0 and 1>,
    "max_platform_fpr_disparity": <float between 0 and 1>
  }
- Target performance is at least `auroc >= 0.80` and `max_platform_fpr_disparity <= 0.25`.
- Produce final artifacts in `/app/artifacts/` and `/app/reports/` so the verifier can inspect them, including at least:
  - `/app/artifacts/metrics.json`
  - `/app/artifacts/predictions.csv`
  - `/app/artifacts/dataset_manifest.json`
  - `/app/artifacts/final_results.json`
  - `/app/models/trained_model.pkl`
  - `/app/reports/report.md`
- Keep the report brief but concrete: summarize the data, the model, the fairness audit, and the final results.
- Ensure the output is reproducible and stable across runs.
