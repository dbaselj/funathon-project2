"""Exercise 2 runner for Project 2 (supervised learning chapter).

Run with:
    uv run exercise2_setup.py
"""

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers.value_encoder import ValueEncoder

DATA_URL = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"


def main() -> None:
    df = pl.read_parquet(DATA_URL)

    # Q1: split into train/val/test (70/15/15)
    train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=42)

    X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
    X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
    X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Arrays -> X_train:{X_train.shape} y_train:{y_train.shape} X_val:{X_val.shape} y_val:{y_val.shape} X_test:{X_test.shape} y_test:{y_test.shape}")

    # Q2: fit label encoder on train labels only
    encoder = LabelEncoder()
    encoder.fit(train_df["code"].to_numpy())

    all_codes = set(df["code"])
    train_codes = set(train_df["code"])
    missing = all_codes - train_codes

    if missing:
        print(f"WARNING: {len(missing)} code(s) missing from training set.")
    else:
        print(f"OK - all {len(all_codes)} codes appear in the training set.")

    # Q3: create ValueEncoder for torchTextClassifiers
    value_encoder = ValueEncoder(label_encoder=encoder)
    print("ValueEncoder ready:", type(value_encoder).__name__)
    print("Encoded classes:", len(encoder.classes_))


if __name__ == "__main__":
    main()
