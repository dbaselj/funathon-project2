"""Exercise 1 runner for Project 2 (supervised learning chapter).

Run with:
    uv run exercise1_setup.py
"""

import mlflow
import polars as pl
from dotenv import load_dotenv

DATA_URL = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"


def main() -> None:
    # Q1: load MLflow credentials from .env
    load_dotenv(override=True)

    tracking_uri = mlflow.get_tracking_uri()
    print(f"MLflow tracking URI: {tracking_uri}")

    # Q2: load dataset
    df = pl.read_parquet(DATA_URL)
    print(df.head())
    print(f"Total rows: {len(df)}")

    # Q3: number of classes
    n_classes = df["code"].n_unique()
    print(f"Number of unique NACE codes: {n_classes}")


if __name__ == "__main__":
    main()
