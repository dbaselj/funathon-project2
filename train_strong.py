"""Stronger supervised training for NACE classifier with MLflow logging.

Run with:
    uv run train_strong.py
"""

import mlflow
import polars as pl
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers.value_encoder import ValueEncoder

DATA_URL = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"


def main() -> None:
    load_dotenv(override=True)
    df = pl.read_parquet(DATA_URL)
    n_classes = df["code"].n_unique()

    train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=42)

    X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
    X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
    X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

    encoder = LabelEncoder()
    encoder.fit(train_df["code"].to_numpy())
    value_encoder = ValueEncoder(label_encoder=encoder)

    tokenizer = WordPieceTokenizer(vocab_size=8000, output_dim=20)
    tokenizer.train(X_train)

    model_config = ModelConfig(embedding_dim=192, num_classes=n_classes)
    ttc = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=model_config,
        value_encoder=value_encoder,
    )

    mlflow.set_experiment("funathon-2026-project2")
    mlflow.pytorch.autolog()
    training_config = TrainingConfig(
        num_epochs=12,
        batch_size=128,
        lr=3e-4,
        patience_early_stopping=3,
    )

    with mlflow.start_run(run_name="strong-training-12ep") as run:
        ttc.train(
            X_train,
            y_train,
            training_config=training_config,
            X_val=X_val,
            y_val=y_val,
            verbose=True,
        )
        mlflow.log_artifacts(training_config.save_path, artifact_path="model_artifacts")

        results_test = ttc.predict(X_test, top_k=1)
        preds = results_test["prediction"].squeeze(1)
        accuracy = (preds == y_test).mean()
        mlflow.log_metric("test_accuracy", float(accuracy))

        print("run_id:", run.info.run_id)
        print("test_accuracy:", f"{accuracy:.4f}")
        print(
            "run_url:",
            f"{mlflow.get_tracking_uri().rstrip('/')}/#/experiments/1/runs/{run.info.run_id}",
        )


if __name__ == "__main__":
    main()
