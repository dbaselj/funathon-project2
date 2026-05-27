"""Exercise 5 runner: load model from MLflow, predict, explain, evaluate.

Run with:
    uv run exercise5_setup.py
"""

import random

import mlflow
import polars as pl
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torchTextClassifiers import torchTextClassifiers
from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char,
    map_attributions_to_word,
)

DATA_URL = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
EXPERIMENT_NAME = "funathon-2026-project2"


def main() -> None:
    load_dotenv(override=True)

    # Rebuild test split exactly like previous exercises.
    df = pl.read_parquet(DATA_URL)
    train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
    _val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=42)
    _x_train, _y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
    X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

    # Load latest run from this experiment and fetch model artifacts.
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {EXPERIMENT_NAME}")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"No runs found in experiment: {EXPERIMENT_NAME}")

    run_id = runs.iloc[0]["run_id"]
    print("Using run_id:", run_id)

    local_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model_artifacts"
    )
    ttc = torchTextClassifiers.load(local_dir)
    ttc.pytorch_model.eval()

    # Q1: top-5 predictions with confidences.
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

    # Q2: attribution extraction at word/char level.
    text_idx = 0
    top_k_idx = 0
    text_sample = example_texts[text_idx]
    offsets = results["offset_mapping"][text_idx]
    word_ids = results["word_ids"][text_idx]
    predicted_code = results["prediction"][text_idx][top_k_idx]
    attributions = results["captum_attributions"][text_idx][top_k_idx]

    words, word_attributions = map_attributions_to_word(
        attributions.unsqueeze(0), text_sample, word_ids, offsets
    )
    char_attributions = map_attributions_to_char(
        attributions.unsqueeze(0), offsets, text_sample
    )

    print("\nAttribution sample:")
    print("  Predicted code:", predicted_code)
    print("  First words:", list(words.values())[:8])
    print("  Word attributions shape:", tuple(word_attributions.shape))
    print("  Char attributions length:", len(char_attributions))

    # Q3: accuracy on test set.
    results_test = ttc.predict(X_test, top_k=1)
    preds = results_test["prediction"].squeeze(1)
    accuracy = (preds == y_test).mean()
    print(f"\nTest accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")


if __name__ == "__main__":
    main()
