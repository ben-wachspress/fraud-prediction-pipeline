"""End-to-end pipeline runner."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console

from src.data.ingest import load_transactions, validate_schema
from src.data.preprocess import apply_smote, clean, encode_categoricals, scale_numerics, split
from src.evaluation.metrics import evaluate, print_report
from src.features.engineering import build_features
from src.models.train import save_model, tune_and_train

app = typer.Typer()
console = Console()


@app.command()
def main(config: Path = typer.Option("configs/pipeline.yaml", help="Path to pipeline config")):
    cfg = yaml.safe_load(config.read_text())
    data_cfg = cfg["data"]
    pre_cfg = cfg["preprocessing"]
    train_cfg = cfg["training"]

    console.rule("[bold blue]1. Ingest")
    df = load_transactions(data_cfg["raw_path"])
    validate_schema(df, [data_cfg["target_column"]])

    console.rule("[bold blue]2. Preprocess")
    df = clean(df, drop_columns=pre_cfg.get("drop_columns", []))
    df, encoders = encode_categoricals(df, pre_cfg["categorical_columns"])
    train, val, test = split(
        df,
        target=data_cfg["target_column"],
        test_size=data_cfg["test_size"],
        val_size=data_cfg["val_size"],
        random_seed=cfg["pipeline"]["random_seed"],
    )

    console.rule("[bold blue]3. Feature Engineering")
    target = data_cfg["target_column"]
    for split_df in (train, val, test):
        split_df = build_features(split_df, data_cfg["timestamp_column"])

    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_train, scaler = scale_numerics(X_train, pre_cfg["numeric_columns"])
    X_train, y_train = apply_smote(X_train, y_train, pre_cfg["smote_sampling_strategy"])

    X_test = test.drop(columns=[target])
    y_test = test[target]
    X_test, _ = scale_numerics(X_test, pre_cfg["numeric_columns"], scaler=scaler, fit=False)

    console.rule("[bold blue]4. Train")
    model, best_params, cv_score = tune_and_train(
        X_train,
        y_train,
        model_type=train_cfg["models"][0],
        n_trials=train_cfg["n_trials"],
        cv_folds=train_cfg["cv_folds"],
        seed=cfg["pipeline"]["random_seed"],
        experiment_name=train_cfg["mlflow_experiment"],
    )
    console.log(f"Best CV PR-AUC: [bold green]{cv_score:.4f}[/bold green]")

    console.rule("[bold blue]5. Evaluate")
    metrics = evaluate(model, X_test, y_test)
    print_report(metrics)

    console.rule("[bold blue]6. Save")
    save_model(model, list(X_train.columns))
    console.log("[bold green]Pipeline complete.[/bold green] Model saved to models/")


if __name__ == "__main__":
    app()
