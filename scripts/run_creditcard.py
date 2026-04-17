"""
End-to-end walkthrough using the ULB Credit Card Fraud dataset.

Dataset: 284,807 transactions, 492 frauds (0.172%)
Features: Time, Amount, V1-V28 (PCA-transformed for confidentiality)
Target: Class (1=fraud, 0=legitimate)
Source: https://openml.org/d/1597
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

console = Console()

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/raw/creditcard.parquet")
PROC_PATH = Path("data/processed"); PROC_PATH.mkdir(parents=True, exist_ok=True)
FEAT_PATH = Path("data/features"); FEAT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "Class"

NUMERIC_COLS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


# ── Stage 1: Ingest ──────────────────────────────────────────────────────────
console.print(Rule("[bold blue]Stage 1 · Ingest"))

df = pd.read_parquet(DATA_PATH)
# OpenML stores the Class column as a category string "0"/"1" — normalise it
df[TARGET] = df[TARGET].astype(str).str.strip().map({"0": 0, "1": 1}).astype(int)

# Subsample for fast demo: keep all 492 frauds + 9508 legit = 10k rows
fraud = df[df[TARGET] == 1]
legit = df[df[TARGET] == 0].sample(n=9508, random_state=SEED)
df = pd.concat([fraud, legit]).sample(frac=1, random_state=SEED).reset_index(drop=True)

t = Table(title="Dataset Overview")
t.add_column("Stat"); t.add_column("Value", style="green")
t.add_row("Rows", f"{len(df):,}")
t.add_row("Columns", str(df.shape[1]))
t.add_row("Fraud transactions", f"{df[TARGET].sum():,}")
t.add_row("Legitimate transactions", f"{(df[TARGET]==0).sum():,}")
t.add_row("Fraud rate", f"{df[TARGET].mean()*100:.3f}%")
t.add_row("Null values", str(df.isnull().sum().sum()))
console.print(t)


# ── Stage 2: Preprocess ──────────────────────────────────────────────────────
console.print(Rule("[bold blue]Stage 2 · Preprocess"))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Clean
df = df.drop_duplicates()
console.log(f"After dedup: {len(df):,} rows")

# Split before any resampling (prevent data leakage)
X = df[NUMERIC_COLS]
y = df[TARGET]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=SEED  # 0.125 * 0.8 = 0.1
)

console.log(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
console.log(f"Train fraud rate: {y_train.mean()*100:.3f}%")
console.log(f"Test fraud rate:  {y_test.mean()*100:.3f}%")

# Scale (fit on train only)
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=NUMERIC_COLS)
X_val_sc   = pd.DataFrame(scaler.transform(X_val),   columns=NUMERIC_COLS)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),  columns=NUMERIC_COLS)

# SMOTE on train only
console.log("Applying SMOTE (sampling_strategy=0.1)...")
sm = SMOTE(sampling_strategy=0.1, random_state=SEED)
X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)
X_train_res = pd.DataFrame(X_train_res, columns=NUMERIC_COLS)

console.log(f"After SMOTE — train size: {len(X_train_res):,}")
console.log(f"After SMOTE — fraud rate: {y_train_res.mean()*100:.2f}%")


# ── Stage 3: Features ────────────────────────────────────────────────────────
console.print(Rule("[bold blue]Stage 3 · Feature Engineering"))

# This dataset is already PCA-transformed; we add a few interaction features
def add_engineered(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    # Amount magnitude bins
    X["amount_log1p"] = np.log1p(np.abs(X["Amount"]))
    # Time-of-day proxy (seconds mod 86400)
    X["time_of_day"] = X["Time"] % 86400
    X["is_night"] = ((X["time_of_day"] < 21600) | (X["time_of_day"] > 79200)).astype(int)
    # Top fraud-discriminating V features combined
    X["v_fraud_signal"] = X[["V4", "V11", "V12", "V14", "V17"]].mean(axis=1)
    return X

X_train_feat = add_engineered(X_train_res)
X_val_feat   = add_engineered(X_val_sc)
X_test_feat  = add_engineered(X_test_sc)

FEATURE_NAMES = list(X_train_feat.columns)
console.log(f"Feature count: {len(FEATURE_NAMES)}  (added amount_log1p, time_of_day, is_night, v_fraud_signal)")


# ── Stage 4: Train ───────────────────────────────────────────────────────────
console.print(Rule("[bold blue]Stage 4 · Train  (XGBoost + LightGBM + Optuna)"))

import optuna
import xgboost as xgb
import lightgbm as lgb
import mlflow, mlflow.sklearn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
mlflow.set_experiment("creditcard-fraud")

def cv_pr_auc(model, X, y, folds=3):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    scores = []
    for tr, val in skf.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[val])[:, 1]
        scores.append(average_precision_score(y.iloc[val], p))
    return float(np.mean(scores))

best_models = {}

for model_type in ("xgboost", "lightgbm"):
    console.log(f"Tuning [bold]{model_type}[/bold] (30 trials)...")

    def objective(trial):
        if model_type == "xgboost":
            m = xgb.XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 150),
                max_depth=trial.suggest_int("max_depth", 3, 7),
                learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                scale_pos_weight=trial.suggest_float("spw", 1, 50),
                eval_metric="aucpr", use_label_encoder=False,
                random_state=SEED, n_jobs=-1,
            )
        else:
            m = lgb.LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 150),
                max_depth=trial.suggest_int("max_depth", 3, 7),
                learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                is_unbalance=True,
                random_state=SEED, n_jobs=-1, verbosity=-1,
            )
        return cv_pr_auc(m, X_train_feat, y_train_res)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, show_progress_bar=True)

    best_p = study.best_params
    console.log(f"  Best CV PR-AUC: [green]{study.best_value:.4f}[/green]  params: {best_p}")

    with mlflow.start_run(run_name=f"{model_type}-best"):
        mlflow.log_params(best_p)
        mlflow.log_metric("cv_pr_auc", study.best_value)

        if model_type == "xgboost":
            model = xgb.XGBClassifier(
                **{k: v for k, v in best_p.items()},
                eval_metric="aucpr", use_label_encoder=False,
                random_state=SEED, n_jobs=-1,
            )
        else:
            model = lgb.LGBMClassifier(
                **{k: v for k, v in best_p.items()},
                is_unbalance=True, random_state=SEED, n_jobs=-1, verbosity=-1,
            )

        model.fit(X_train_feat, y_train_res)
        val_pr = average_precision_score(y_val, model.predict_proba(X_val_feat)[:, 1])
        mlflow.log_metric("val_pr_auc", val_pr)
        mlflow.sklearn.log_model(model, "model")
        console.log(f"  Val PR-AUC: [cyan]{val_pr:.4f}[/cyan]")

    best_models[model_type] = (model, study.best_value, val_pr)


# ── Stage 5: Evaluate ────────────────────────────────────────────────────────
console.print(Rule("[bold blue]Stage 5 · Evaluate on Hold-out Test Set"))

from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve,
)

results_table = Table(title="Model Comparison")
for col in ("Model", "Test PR-AUC", "Test ROC-AUC", "Best F1", "Threshold", "Precision", "Recall"):
    results_table.add_column(col)

best_overall_model = None
best_overall_pr = 0.0

for name, (model, cv_pr, val_pr) in best_models.items():
    proba = model.predict_proba(X_test_feat)[:, 1]

    # Find threshold maximising F1
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, proba)
    f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-9)
    best_idx = np.argmax(f1_curve[:-1])
    threshold = float(thresholds[best_idx])

    preds = (proba >= threshold).astype(int)
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, preds)
    prec = precision_curve[best_idx]
    rec  = recall_curve[best_idx]

    results_table.add_row(
        name,
        f"{pr_auc:.4f}",
        f"{roc_auc:.4f}",
        f"{f1:.4f}",
        f"{threshold:.3f}",
        f"{prec:.3f}",
        f"{rec:.3f}",
    )

    if pr_auc > best_overall_pr:
        best_overall_pr = pr_auc
        best_overall_model = (name, model, threshold, proba, preds)

console.print(results_table)

# Confusion matrix for best model
best_name, best_model, best_thresh, best_proba, best_preds = best_overall_model
cm = confusion_matrix(y_test, best_preds)
console.print(f"\n[bold]Confusion Matrix — {best_name}[/bold]  (threshold={best_thresh:.3f})")
console.print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
console.print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
console.print(f"\n[bold]Classification Report:[/bold]")
console.print(classification_report(y_test, best_preds, target_names=["Legit","Fraud"]))

# Feature importance
fi = pd.Series(best_model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
top10 = Table(title=f"Top 10 Feature Importances — {best_name}")
top10.add_column("Feature", style="cyan"); top10.add_column("Importance", style="green")
for feat, imp in fi.head(10).items():
    top10.add_row(feat, f"{imp:.4f}")
console.print(top10)


# ── Stage 6: Save ────────────────────────────────────────────────────────────
console.print(Rule("[bold blue]Stage 6 · Save Artifacts"))

joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
(MODEL_DIR / "feature_names.json").write_text(json.dumps(FEATURE_NAMES))
(MODEL_DIR / "threshold.json").write_text(json.dumps({"threshold": best_thresh, "model": best_name}))

console.print(f"[green]Saved:[/green] models/best_model.pkl  ({best_name})")
console.print(f"[green]Saved:[/green] models/scaler.pkl")
console.print(f"[green]Saved:[/green] models/feature_names.json  ({len(FEATURE_NAMES)} features)")
console.print(f"[green]Saved:[/green] models/threshold.json  (threshold={best_thresh:.3f})")


# ── Stage 7: Inference demo ──────────────────────────────────────────────────
console.print(Rule("[bold blue]Stage 7 · Inference Demo"))

# Pick 3 real frauds and 3 real legit transactions from test set
fraud_samples   = X_test[y_test == 1].head(3)
legit_samples   = X_test[y_test == 0].head(3)
demo_X_raw = pd.concat([fraud_samples, legit_samples])
demo_y     = pd.concat([y_test[y_test == 1].head(3), y_test[y_test == 0].head(3)])

demo_X_sc   = pd.DataFrame(scaler.transform(demo_X_raw[NUMERIC_COLS]), columns=NUMERIC_COLS)
demo_X_feat = add_engineered(demo_X_sc)

demo_proba = best_model.predict_proba(demo_X_feat)[:, 1]
demo_pred  = (demo_proba >= best_thresh).astype(int)

demo_table = Table(title="Inference on 6 Real Transactions (3 fraud + 3 legit)")
demo_table.add_column("True Label")
demo_table.add_column("Fraud Prob")
demo_table.add_column("Prediction")
demo_table.add_column("Correct?")

for true_lbl, prob, pred in zip(demo_y, demo_proba, demo_pred):
    correct = "✓" if pred == true_lbl else "✗"
    style = "green" if pred == true_lbl else "red"
    demo_table.add_row(
        "FRAUD" if true_lbl == 1 else "legit",
        f"{prob:.4f}",
        "FRAUD" if pred == 1 else "legit",
        f"[{style}]{correct}[/{style}]",
    )

console.print(demo_table)
console.print(Rule("[bold green]Pipeline complete"))
