# Fraud Prediction Pipeline

End-to-end ML pipeline for credit card fraud detection, covering data ingestion, feature engineering, model training, evaluation, and serving.

## Project Structure

```
fraud-prediction-pipeline/
├── data/
│   ├── raw/            # Raw input data (gitignored)
│   ├── processed/      # Cleaned/transformed data
│   └── features/       # Feature store outputs
├── notebooks/          # Exploratory analysis
├── src/
│   ├── data/           # Data ingestion & preprocessing
│   ├── features/       # Feature engineering
│   ├── models/         # Model training & hyperparameter tuning
│   ├── evaluation/     # Metrics, plots, reports
│   └── serving/        # Inference API
├── tests/
│   ├── unit/
│   └── integration/
├── configs/            # YAML configs for pipeline stages
├── scripts/            # CLI entry points
└── .github/workflows/  # CI/CD
```

## Quickstart

```bash
# Install dependencies
pip install -e ".[dev]"

# Run full pipeline
python scripts/run_pipeline.py --config configs/pipeline.yaml

# Train model only
python scripts/train.py --config configs/train.yaml

# Start inference API
uvicorn src.serving.app:app --reload
```

## Pipeline Stages

1. **Ingest** — load raw transaction data
2. **Preprocess** — clean, encode, handle class imbalance (SMOTE)
3. **Feature engineering** — velocity features, aggregations, embeddings
4. **Train** — XGBoost + LightGBM with Optuna hyperparameter search
5. **Evaluate** — PR-AUC, ROC-AUC, confusion matrix, threshold analysis
6. **Serve** — FastAPI inference endpoint with feature validation

## Key Metrics

Fraud detection prioritizes **Precision-Recall AUC** over accuracy due to severe class imbalance (~0.17% fraud rate in typical datasets).
