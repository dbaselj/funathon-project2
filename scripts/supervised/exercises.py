"""Supervised learning exercises 1-5 for NACE classification.

Run with:
    uv run exercises.py
"""

import random

import mlflow
import polars as pl
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char,
    map_attributions_to_word,
)
from torchTextClassifiers.value_encoder import ValueEncoder

DATA_URL = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
EXPERIMENT_NAME = "funathon-2026-project2"


def main() -> None:
    load_dotenv(override=True)

    # ── Exercise 1: data exploration ──────────────────────────────────────────
    print("=== Exercise 1: Data exploration ===")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    df = pl.read_parquet(DATA_URL)
    print(df.head())
    print(f"Total rows: {len(df)}")
    n_classes = df["code"].n_unique()
    print(f"Number of unique NACE codes: {n_classes}")

    # ── Exercise 2: split + encoders ─────────────────────────────────────────
    print("\n=== Exercise 2: Split + encoders ===")
    train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=42)

    X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
    X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
    X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    encoder = LabelEncoder()
    encoder.fit(y_train)
    value_encoder = ValueEncoder(label_encoder=encoder)

    missing = set(df["code"]) - set(train_df["code"])
    if missing:
        print(f"WARNING: {len(missing)} code(s) missing from training set.")
    else:
        print(f"OK — all {n_classes} codes appear in the training set.")
    print("Encoded classes:", len(encoder.classes_))

    # ── Exercise 3: tokenizer ─────────────────────────────────────────────────
    print("\n=== Exercise 3: Tokenizer ===")
    tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)
    tokenizer.train(X_train)

    sample = X_train[0]
    tokenized = tokenizer.tokenize(sample)
    token_ids = tokenized.input_ids.squeeze(0)
    print("Output tensor size:", tokenized.input_ids.shape)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Raw text:", sample)
    print("Tokens:", tokenizer.tokenizer.convert_ids_to_tokens(token_ids))

    # ── Exercise 4: train model ───────────────────────────────────────────────
    print("\n=== Exercise 4: Training ===")
    model_config = ModelConfig(embedding_dim=96, num_classes=n_classes)
    ttc = torchTextClassifiers(
        tokenizer=tokenizer, model_config=model_config, value_encoder=value_encoder
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.pytorch.autolog()
    training_config = TrainingConfig(
        num_epochs=100, batch_size=128, lr=5e-4, patience_early_stopping=5
    )

    with mlflow.start_run() as run:
        ttc.train(
            X_train, y_train,
            training_config=training_config,
            X_val=X_val, y_val=y_val,
            verbose=True,
        )
        mlflow.log_artifacts(training_config.save_path, artifact_path="model_artifacts")
        run_id = run.info.run_id
        print("run_id:", run_id)
        print("run_url:", f"{mlflow.get_tracking_uri().rstrip('/')}/#/experiments/1/runs/{run_id}")

    # ── Exercise 5: predict, explain, evaluate ────────────────────────────────
    print("\n=== Exercise 5: Predict + explain + evaluate ===")
    random_indices = random.sample(range(len(X_test)), 3)
    example_texts = X_test[random_indices]
    example_true_codes = y_test[random_indices]
    top_k = 5

    results = ttc.predict(example_texts, top_k=top_k, explain_with_captum=True)
    for i, text in enumerate(example_texts):
        predicted_codes = [results["prediction"][i][k] for k in range(top_k)]
        confidence = [results["confidence"][i][k].item() for k in range(top_k)]
        print(f"\nText: {text}")
        print(f"  True code: {example_true_codes[i]}")
        for code, conf in zip(predicted_codes, confidence):
            print(f"  {code} (confidence: {conf:.3f})")

    offsets = results["offset_mapping"][0]
    word_ids = results["word_ids"][0]
    attributions = results["captum_attributions"][0][0]
    words, word_attributions = map_attributions_to_word(
        attributions.unsqueeze(0), example_texts[0], word_ids, offsets
    )
    char_attributions = map_attributions_to_char(
        attributions.unsqueeze(0), offsets, example_texts[0]
    )
    print("\nAttribution sample:")
    print("  Predicted code:", results["prediction"][0][0])
    print("  First words:", list(words.values())[:8])
    print("  Word attributions shape:", tuple(word_attributions.shape))
    print("  Char attributions length:", len(char_attributions))

    results_test = ttc.predict(X_test, top_k=1)
    preds = results_test["prediction"].squeeze(1)
    accuracy = (preds == y_test).mean()
    print(f"\nTest accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")


if __name__ == "__main__":
    main()
