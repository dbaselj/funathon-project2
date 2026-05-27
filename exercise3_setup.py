"""Exercise 3 runner for Project 2 (supervised learning chapter).

Run with:
    uv run exercise3_setup.py
"""

import polars as pl
from sklearn.model_selection import train_test_split
from torchTextClassifiers.tokenizers import WordPieceTokenizer

DATA_URL = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"


def main() -> None:
    df = pl.read_parquet(DATA_URL)

    # Recreate X_train exactly like Exercise 2 (70/15/15 split).
    train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
    _val_df, _test_df = train_test_split(tmp_df, test_size=0.50, random_state=42)
    X_train = train_df["label"].to_numpy()

    tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)
    tokenizer.train(X_train)

    sample = X_train[0]
    tokenized = tokenizer.tokenize(sample)
    token_ids = tokenized.input_ids.squeeze(0)

    print("Output tensor size:", tokenized.input_ids.shape)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Raw text:", sample)
    print("Tokens id:", token_ids)
    print("Tokens:", tokenizer.tokenizer.convert_ids_to_tokens(token_ids))


if __name__ == "__main__":
    main()
