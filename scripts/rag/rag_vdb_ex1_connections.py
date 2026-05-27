"""RAG VDB Exercise 1: connections (llm.lab + Qdrant)."""

import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient


def _optional_env_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc


def _qdrant_client_from_env() -> QdrantClient:
    kwargs = {
        "url": os.environ["QDRANT_URL"],
        "api_key": os.environ["QDRANT_API_KEY"],
        "check_compatibility": False,
    }
    port = _optional_env_int("QDRANT_API_PORT")
    if port is not None:
        kwargs["port"] = port
    return QdrantClient(**kwargs)


def main() -> None:
    load_dotenv(override=True)

    required = [
        "LLMLAB_URL",
        "LLMLAB_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
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
    client_qdrant = _qdrant_client_from_env()
    collections = client_qdrant.get_collections()
    print("\nQdrant collections:")
    if not collections.collections:
        print("- (none)")
    for c in collections.collections:
        print(f"- {c.name}")


if __name__ == "__main__":
    main()
