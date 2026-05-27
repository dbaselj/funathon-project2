"""RAG exercises for NACE 2.1 classification.

Phase 1 — Vector database construction (VDB exercises 1-3):
  - Connect to Qdrant and llm.lab
  - Structure NACE documents
  - Embed and upload to Qdrant

Phase 2 — Pipeline execution and evaluation (Gen exercises 1-3):
  - Classify a single activity end-to-end
  - Batch inference on 100 samples
  - Evaluate retriever and pipeline accuracy

Run with:
    uv run rag.py
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional
from uuid import NAMESPACE_DNS, uuid5

import duckdb
import pandas as pd
from dotenv import load_dotenv
from more_itertools import chunked
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

PATH_NACE = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/NACE_Rev2.1_Structure_Explanatory_Notes_EN.tsv"
EMB_MODEL_NAME = "qwen3-embedding-8b"
GEN_MODEL_NAME = "gemma4-26b-moe"
EMB_DIM = 4096
COLLECTION_NAME = "nace-collection"
BATCH_SIZE = 16
NACE_NAMESPACE = uuid5(NAMESPACE_DNS, "nace-rev2")
RETRIEVER_LIMIT = 5
TEMPERATURE = 0.1
SAMPLE_SIZE = 100

SYSTEM_PROMPT = """\
You are an expert classifier for the NACE 2.1 nomenclature (Statistical Classification of Economic Activities in the European Community).
Given a company activity description and a short list of candidate NACE codes, your job is to pick the single most appropriate code from the candidates — or to declare the activity not codable if the description is too ambiguous.
Always reply with a valid JSON object matching the requested schema. No explanations, no extra text.
"""

USER_PROMPT_TEMPLATE = """\
## Activity to classify
{activity}

## Candidate NACE codes and their explanatory notes
{proposed_nace_descriptions}

## Rules
- Pick exactly one code from this list: [{proposed_nace_codes}]. Do not invent codes outside the list.
- If several activities are mentioned, only consider the first one.
- If the description is too vague to decide, return `nace_code: null` and `codable: false`.

## Output — valid JSON only
{{
  "nace_code": "<one code from the candidate list, or null>",
  "codable": <true | false>,
  "confidence": <float between 0.0 and 1.0>
}}
"""


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


# ── RAG pipeline ──────────────────────────────────────────────────────────────

class NaceClassificationResult(BaseModel):
    nace_code: Optional[str] = Field(description="Chosen NACE code from the candidate list, or null")
    codable: bool = Field(description="False if the description is too vague to code")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")


def run_rag_pipeline(activity: str, client_llm: OpenAI, client_qdrant: QdrantClient) -> dict:
    embedding = client_llm.embeddings.create(model=EMB_MODEL_NAME, input=activity).data[0].embedding

    points = client_qdrant.query_points(
        collection_name=COLLECTION_NAME, query=embedding, limit=RETRIEVER_LIMIT
    )
    descriptions, codes = [], []
    for point in points.model_dump()["points"]:
        descriptions.append(point["payload"]["text"])
        codes.append(point["payload"]["code"])

    user_prompt = USER_PROMPT_TEMPLATE.format(
        activity=activity,
        proposed_nace_descriptions="## " + "\n\n## ".join(descriptions),
        proposed_nace_codes=", ".join(codes),
    )
    response = client_llm.chat.completions.parse(
        model=GEN_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        response_format=NaceClassificationResult,
    )
    result = response.choices[0].message.parsed.model_dump()
    result["retrieved_codes"] = codes
    return result


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

    # ── VDB Exercise 1: connections ───────────────────────────────────────────
    print("=== VDB Exercise 1: Connections ===")
    print("llm.lab models:")
    for model in client_llm.models.list().data:
        print(f"  - {model.id}")

    collections = client_qdrant.get_collections()
    print("\nQdrant collections:")
    for c in collections.collections:
        print(f"  - {c.name}")
    if not collections.collections:
        print("  (none)")

    # ── VDB Exercise 2: NACE document structure ───────────────────────────────
    print("\n=== VDB Exercise 2: NACE document structure ===")
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

    # ── VDB Exercise 3: build vector store ───────────────────────────────────
    print("\n=== VDB Exercise 3: Build Qdrant vector store ===")
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

    # ── Gen Exercise 1: single example ───────────────────────────────────────
    print("\n=== Gen Exercise 1: Single example ===")
    activity = "Installation, maintenance and repair of residential air conditioning systems for private customers"
    print("Activity:", activity)

    embedding = client_llm.embeddings.create(model=EMB_MODEL_NAME, input=activity).data[0].embedding
    print("Embedding length:", len(embedding))

    points = client_qdrant.query_points(
        collection_name=COLLECTION_NAME, query=embedding, limit=RETRIEVER_LIMIT
    )
    codes = [p["payload"]["code"] for p in points.model_dump()["points"]]
    print("Retrieved codes:", codes)

    result = run_rag_pipeline(activity, client_llm, client_qdrant)
    print("\nLLM result:")
    print(json.dumps({k: v for k, v in result.items() if k != "retrieved_codes"}, indent=2))

    # ── Gen Exercises 2 + 3: batch inference + evaluation ────────────────────
    print("\n=== Gen Exercises 2 + 3: Batch inference + evaluation ===")
    con = duckdb.connect(database=":memory:")
    _load_httpfs(con)
    annotations = con.sql(
        f"""
        SELECT * FROM read_parquet(
            'https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet'
        ) USING SAMPLE {SAMPLE_SIZE}
        """
    ).to_df().to_dict(orient="records")
    print(f"Dataset loaded: {len(annotations)} rows")

    records = []
    for row in tqdm(annotations, desc="Coding"):
        try:
            pred = run_rag_pipeline(row["label"], client_llm, client_qdrant)
        except Exception as e:
            pred = {"nace_code": None, "codable": False, "confidence": 0.0, "retrieved_codes": []}
            print(f"Error for '{row['label'][:60]}': {e}")

        records.append({
            "activity": row["label"],
            "true_code": row["code"],
            "pred_code": pred.get("nace_code"),
            "codable": pred.get("codable", False),
            "confidence": pred.get("confidence", 0.0),
            "retrieved_codes": pred.get("retrieved_codes", []),
        })

    results = pd.DataFrame(records)
    results["retriever_hit"] = results.apply(lambda r: r["true_code"] in r["retrieved_codes"], axis=1)
    results["pipeline_correct"] = results["pred_code"] == results["true_code"]

    retriever_acc = results["retriever_hit"].mean()
    llm_cond_acc = results.loc[results["retriever_hit"], "pipeline_correct"].mean()
    pipeline_acc = results["pipeline_correct"].mean()

    print("\nMetrics:")
    print(f"Retriever@{RETRIEVER_LIMIT} accuracy: {retriever_acc:.1%}")
    print(f"LLM conditional accuracy:  {llm_cond_acc:.1%}")
    print(f"Pipeline accuracy:         {pipeline_acc:.1%}")


if __name__ == "__main__":
    main()
