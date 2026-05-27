"""RAG VDB Exercise 1: connections (llm.lab + Qdrant)."""

import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient


def main() -> None:
    load_dotenv(override=True)

    required = [
        "LLMLAB_URL",
        "LLMLAB_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "QDRANT_API_PORT",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print("Missing env vars:", ", ".join(missing))
        return

    # Question 1 + 2: llm.lab client and available models
    client_llmlab = OpenAI(
        base_url=os.environ["LLMLAB_URL"],
        api_key=os.environ["LLMLAB_API_KEY"],
    )
    print("llm.lab models:")
    for model in client_llmlab.models.list().data:
        print(f"- {model.id}")

    # Question 3: Qdrant client and list collections
    client_qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        port=os.environ["QDRANT_API_PORT"],
        check_compatibility=False,
    )
    collections = client_qdrant.get_collections()
    print("\nQdrant collections:")
    if not collections.collections:
        print("- (none)")
    for c in collections.collections:
        print(f"- {c.name}")


if __name__ == "__main__":
    main()
