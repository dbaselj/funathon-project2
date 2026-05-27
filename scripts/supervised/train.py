"""Supervised training for NACE classification.

Stratified split, minority upsampling, 12 epochs.
Logs accuracy, macro-F1, and top-3 accuracy to MLflow.

Run with:
    uv run train.py
"""

import mlflow
import numpy as np
import polars as pl
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers.value_encoder import ValueEncoder

DATA_URL = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
EXPERIMENT_NAME = "funathon-2026-project2"


def stratified_split(df: pl.DataFrame):
    labels = df["code"].to_numpy()
    idx = np.arange(len(df))
    train_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=42, stratify=labels)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.50, random_state=42, stratify=labels[tmp_idx]
    )
    return df[train_idx], df[val_idx], df[test_idx]


def upsample_minority(train_df: pl.DataFrame) -> pl.DataFrame:
    counts_df = train_df.group_by("code").len()
    max_count = int(counts_df["len"].max())
    parts = []
    for row in counts_df.iter_rows(named=True):
        cls_rows = train_df.filter(pl.col("code") == row["code"])
        count = int(row["len"])
        if count < max_count:
            extra_idx = np.random.choice(count, size=max_count - count, replace=True)
            cls_rows = pl.concat([cls_rows, cls_rows[extra_idx]], how="vertical")
        parts.append(cls_rows)
    return pl.concat(parts, how="vertical").sample(fraction=1.0, shuffle=True, seed=42)


def main() -> None:
    load_dotenv(override=True)
    np.random.seed(42)

    df = pl.read_parquet(DATA_URL)
    train_df, val_df, test_df = stratified_split(df)
    train_df_balanced = upsample_minority(train_df)

    X_train = train_df_balanced["label"].to_numpy()
    y_train = train_df_balanced["code"].to_numpy()
    X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
    X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

    encoder = LabelEncoder()
    encoder.fit(train_df["code"].to_numpy())
    value_encoder = ValueEncoder(label_encoder=encoder)

    tokenizer = WordPieceTokenizer(vocab_size=8000, output_dim=20)
    tokenizer.train(X_train)

    model_config = ModelConfig(embedding_dim=192, num_classes=df["code"].n_unique())
    ttc = torchTextClassifiers(
        tokenizer=tokenizer, model_config=model_config, value_encoder=value_encoder
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.pytorch.autolog()
    training_config = TrainingConfig(
        num_epochs=12, batch_size=128, lr=3e-4, patience_early_stopping=3
    )

    with mlflow.start_run(run_name="train-stratified-balanced") as run:
        ttc.train(
            X_train, y_train,
            training_config=training_config,
            X_val=X_val, y_val=y_val,
            verbose=True,
        )
        mlflow.log_artifacts(training_config.save_path, artifact_path="model_artifacts")

        pred1 = ttc.predict(X_test, top_k=1)["prediction"].squeeze(1)
        pred3 = ttc.predict(X_test, top_k=3)["prediction"]

        acc = accuracy_score(y_test, pred1)
        macro_f1 = f1_score(y_test, pred1, average="macro")
        top3 = np.mean([y_test[i] in pred3[i] for i in range(len(y_test))])

        mlflow.log_metric("test_accuracy", float(acc))
        mlflow.log_metric("test_macro_f1", float(macro_f1))
        mlflow.log_metric("test_top3_accuracy", float(top3))

        print("run_id:", run.info.run_id)
        print("test_accuracy:", f"{acc:.4f}")
        print("test_macro_f1:", f"{macro_f1:.4f}")
        print("test_top3_accuracy:", f"{top3:.4f}")
        print("run_url:", f"{mlflow.get_tracking_uri().rstrip('/')}/#/experiments/1/runs/{run.info.run_id}")


if __name__ == "__main__":
    main()
