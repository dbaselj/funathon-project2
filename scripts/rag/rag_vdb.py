"""RAG vector database pipeline: connect, structure NACE docs, build Qdrant store.

Covers VDB exercises 1-3.

Run with:
    uv run rag_vdb.py
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional
from uuid import NAMESPACE_DNS, uuid5

import duckdb
from dotenv import load_dotenv
from more_itertools import chunked
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

PATH_NACE = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/NACE_Rev2.1_Structure_Explanatory_Notes_EN.tsv"
EMB_MODEL_NAME = "qwen3-embedding-8b"
EMB_DIM = 4096
COLLECTION_NAME = "nace-collection"
BATCH_SIZE = 16
NACE_NAMESPACE = uuid5(NAMESPACE_DNS, "nace-rev2")


# ── Shared helpers ────────────────────────────────────────────────────────────

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


def _load_httpfs(con: duckdb.DuckDBPyConnection) -> None:
    try:
        con.execute("LOAD httpfs;")
    except duckdb.Error:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    cleaned = " ".join(str(value).replace("\n", " ").split())
    return cleaned or None


# ── NaceDocument ──────────────────────────────────────────────────────────────

@dataclass
class NaceDocument:
    code: str
    heading: str
    level: int
    parent_code: Optional[str] = None
    includes: Optional[str] = None
    includes_also: Optional[str] = None
    excludes: Optional[str] = None

    text: str = field(init=False)
    vector: Optional[List[float]] = field(default=None, init=False)

    @classmethod
    def from_raw(
        cls,
        raw: dict,
        with_includes_also: bool = True,
        with_excludes: bool = True,
    ) -> "NaceDocument":
        for key in ("CODE", "HEADING", "LEVEL"):
            if not raw.get(key):
                raise ValueError(f"Missing required field: {key}")

        level = int(raw["LEVEL"])
        if not (1 <= level <= 4):
            raise ValueError(f"Invalid level: {level}")

        parent_code = _clean(raw.get("PARENT_CODE"))
        if level > 1 and not parent_code:
            raise ValueError(f"Missing parent code for level {level}: {raw['CODE']}")

        obj = cls(
            code=str(raw["CODE"]).strip(),
            heading=_clean(raw["HEADING"]),
            level=level,
            parent_code=parent_code,
            includes=_clean(raw.get("Includes")),
            includes_also=_clean(raw.get("IncludesAlso")),
            excludes=_clean(raw.get("Excludes")),
        )
        obj.text = obj.to_embedding_text(
            with_includes_also=with_includes_also, with_excludes=with_excludes
        )
        return obj

    def to_embedding_text(
        self,
        *,
        with_includes_also: bool = False,
        with_excludes: bool = False,
    ) -> str:
        parts = [f"# Code: {self.code}", f"# Title: {self.heading}"]
        if self.includes:
            parts.extend(["", "## Includes:", self.includes.strip()])
        if with_includes_also and self.includes_also:
            parts.extend(["", "## Also includes:", self.includes_also.strip()])
        if with_excludes and self.excludes:
            parts.extend(["", "## Excludes:", self.excludes.strip()])
        return "\n".join(parts).replace("\\n", "\n").strip()

    def get_embeddings(self, client: OpenAI, model: str) -> List[float]:
        try:
            response = client.embeddings.create(model=model, input=self.text)
            self.vector = response.data[0].embedding
            return self.vector
        except Exception as e:
            raise RuntimeError(f"Embedding failed for doc {self.code}: {e}") from e

    def to_qdrant_point(self) -> PointStruct:
        if self.vector is None:
            raise ValueError("vector is missing — call get_embeddings() first")
        return PointStruct(
            id=str(uuid5(NACE_NAMESPACE, self.code)),
            vector=self.vector,
            payload={
                "code": self.code,
                "level": self.level,
                "parent_code": self.parent_code,
                "text": self.text,
            },
        )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_nace_rows() -> List[dict]:
    con = duckdb.connect(database=":memory:")
    _load_httpfs(con)
    return (
        con.execute(
            f"SELECT * FROM read_csv('{PATH_NACE}', delim='\t', header=true, all_varchar=true)"
        )
        .to_arrow_table()
        .to_pylist()
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(override=True)

    required = ["LLMLAB_URL", "LLMLAB_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print("Missing env vars:", ", ".join(missing))
        return

    client_llm = OpenAI(base_url=os.environ["LLMLAB_URL"], api_key=os.environ["LLMLAB_API_KEY"])
    client_qdrant = _qdrant_client_from_env()

    # ── Exercise 1: connections ───────────────────────────────────────────────
    print("=== Exercise 1: Connections ===")
    print("llm.lab models:")
    for model in client_llm.models.list().data:
        print(f"  - {model.id}")

    collections = client_qdrant.get_collections()
    print("\nQdrant collections:")
    for c in collections.collections:
        print(f"  - {c.name}")
    if not collections.collections:
        print("  (none)")

    # ── Exercise 2: NACE document structure ───────────────────────────────────
    print("\n=== Exercise 2: NACE document structure ===")
    nace = load_nace_rows()
    print(f"Loaded rows: {len(nace)}")
    print("Sample raw record (index 22):")
    print(nace[22])

    nace_documents = [NaceDocument.from_raw(row) for row in nace]
    print(f"\nBuilt NaceDocument count: {len(nace_documents)}")

    doc = NaceDocument.from_raw(nace[50], with_includes_also=True, with_excludes=True)
    print("\n=== WITH exclusions ===")
    print(doc.text)
    print("\n=== WITHOUT exclusions ===")
    print(doc.to_embedding_text(with_includes_also=True, with_excludes=False))

    # ── Exercise 3: build vector store ───────────────────────────────────────
    print("\n=== Exercise 3: Build Qdrant vector store ===")
    if client_qdrant.collection_exists(COLLECTION_NAME):
        client_qdrant.delete_collection(COLLECTION_NAME)
    client_qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMB_DIM, distance=Distance.COSINE),
    )
    print(f"Collection created: {COLLECTION_NAME}")

    sample_docs = [NaceDocument.from_raw(row) for row in nace[:10]]
    for doc in sample_docs:
        doc.get_embeddings(client_llm, EMB_MODEL_NAME)
    print("Sample embedding length:", len(sample_docs[0].vector or []))
    print("Sample first 8 values:", (sample_docs[0].vector or [])[:8])

    nace_points: List[PointStruct] = []
    for raw in tqdm(nace, desc="Embedding NACE docs", unit="doc"):
        doc = NaceDocument.from_raw(raw)
        doc.get_embeddings(client_llm, EMB_MODEL_NAME)
        nace_points.append(doc.to_qdrant_point())

    point = nace_points[0].model_dump()
    vector = point["vector"]
    point["vector"] = f"[{vector[0]:.4f}, ..., {vector[-1]:.4f}] ({len(vector)} dims)"
    print("\nFirst PointStruct preview:")
    print(json.dumps(point, indent=2, ensure_ascii=False))

    for batch in tqdm(list(chunked(nace_points, BATCH_SIZE)), desc="Uploading", unit="batch"):
        try:
            client_qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
        except Exception as e:
            tqdm.write(f"Batch failed: {e}")

    count = client_qdrant.count(COLLECTION_NAME)
    print(f"\nCollection count: {count.count}")


if __name__ == "__main__":
    main()
