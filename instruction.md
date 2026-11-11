# Fairness-Aware Tabular Classification

You must train a tabular binary classification model on the provided fairness dataset and optimize for both predictive quality and platform-level false-positive rate parity.

## Data
Load the dataset from exactly one of these paths (use the first one that exists):
- `/app/data/data.csv`
- `/app/data/input_data.csv`
- `/app/data/input_data.npz`

## Required Deliverables
Your solution must write the following files upon execution:
1. `/app/artifacts/dataset_manifest.json`: Metadata about the dataset (source, n_samples, n_features, target_column, protected_column, splits).
2. `/app/artifacts/processed_data.npz`: Preprocessed dataset splits.
3. `/app/artifacts/predictions.csv`: Model predictions on the evaluation set.
4. `/app/models/trained_model.pkl`: The serialized trained model.
5. `/app/artifacts/final_results.json`: Summary of final execution metrics.
6. `/app/artifacts/metrics.json`: Performance metrics including `auroc` and `max_platform_fpr_disparity`.
7. `/app/reports/report.md`: A technical report of at least 40 words containing the terms: `model`, `fairness`, `platform`, `auroc`, and `results`.

The target column is named `label` (binary).  
If a column named `platform` exists, treat it as the protected attribute for the fairness audit.  
If the `platform` column does not exist, select the most suitable categorical column in the dataset to be the group attribute for the false-positive rate parity check.

## Success Criteria (must both be met)
- `auroc >= 0.80`
- `max_platform_fpr_disparity <= 0.25`

## Required Metrics File
Write the final metrics to `/app/artifacts/metrics.json` with exactly these keys:

```json
{
  "auroc": <float between 0 and 1>,
  "max_platform_fpr_disparity": <float between 0 and 1>
}