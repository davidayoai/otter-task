# Fairness Audit Report

- Target column: target
- Protected column: platform
- Test AUROC: 0.8157
- Max platform FPR disparity: 0.0143

The model uses a standard logistic-regression pipeline with median imputation, one-hot encoding for categorical features, and standard scaling for numeric features.
This setup keeps the model interpretable while providing stable probability estimates for a fairness audit.
