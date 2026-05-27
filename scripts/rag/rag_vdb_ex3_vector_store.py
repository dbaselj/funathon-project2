"""RAG VDB Exercise 3: create and populate Qdrant vector store."""

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


def _load_httpfs(con: duckdb.DuckDBPyConnection) -> None:
    try:
        con.execute("LOAD httpfs;")
    except duckdb.Error:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")


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


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    cleaned = " ".join(str(value).replace("\n", " ").split())
    return cleaned or None


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

    def get_embeddings(self, client_llmlab: OpenAI, emb_model: str) -> List[float]:
        try:
            response = client_llmlab.embeddings.create(model=emb_model, input=self.text)
            self.vector = response.data[0].embedding
            return self.vector
        except Exception as e:
            raise RuntimeError(f"Embedding failed for doc {self.code}: {str(e)}") from e

    def to_qdrant_point(self) -> PointStruct:
        if self.vector is None:
            raise ValueError("vector is missing")
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


def load_nace_rows() -> List[dict]:
    con = duckdb.connect(database=":memory:")
    _load_httpfs(con)
    table = con.execute(
        f"""
        SELECT *
        FROM read_csv(
            '{PATH_NACE}',
            delim='\t',
            header=true,
            all_varchar=true
        )
        """
    ).to_arrow_table()
    return table.to_pylist()


def main() -> None:
    load_dotenv(override=True)
    client_llmlab = OpenAI(
        base_url=os.environ["LLMLAB_URL"],
        api_key=os.environ["LLMLAB_API_KEY"],
    )
    client_qdrant = _qdrant_client_from_env()

    # Q1: Create collection
    if client_qdrant.collection_exists(collection_name=COLLECTION_NAME):
        client_qdrant.delete_collection(collection_name=COLLECTION_NAME)
    client_qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMB_DIM, distance=Distance.COSINE),
    )
    print(f"Collection created: {COLLECTION_NAME}")

    nace = load_nace_rows()
    print(f"NACE rows loaded: {len(nace)}")

    # Q2 + Q3: sample embeddings inspection
    sample_docs = [
        NaceDocument.from_raw(row, with_includes_also=True, with_excludes=True)
        for row in nace[:10]
    ]
    for doc in sample_docs:
        doc.get_embeddings(client_llmlab, EMB_MODEL_NAME)
    print("Sample embedding length:", len(sample_docs[0].vector or []))
    print("Sample first 8 values:", (sample_docs[0].vector or [])[:8])

    # Q5: full points generation
    nace_points: List[PointStruct] = []
    for raw in tqdm(nace, desc="Embedding NACE docs", unit="doc"):
        doc = NaceDocument.from_raw(raw, with_includes_also=True, with_excludes=True)
        doc.get_embeddings(client_llmlab, EMB_MODEL_NAME)
        nace_points.append(doc.to_qdrant_point())

    # Q4 (inspection)
    point = nace_points[0].model_dump()
    vector = point["vector"]
    point["vector"] = f"[{vector[0]:.4f}, {vector[1]:.4f}, ..., {vector[-1]:.4f}] ({len(vector)} dims)"
    print("\nFirst PointStruct preview:")
    print(json.dumps(point, indent=2, ensure_ascii=False))

    # Q6: batched upsert
    for batch in tqdm(
        list(chunked(nace_points, BATCH_SIZE)),
        desc="Uploading to Qdrant",
        unit="batch",
    ):
        try:
            client_qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
        except Exception as e:
            tqdm.write(f"Batch failed: {e}")

    # Q7: verify count
    count = client_qdrant.count(collection_name=COLLECTION_NAME)
    print("\nCollection count:", count.count)


if __name__ == "__main__":
    main()
