"""RAG generation pipeline: single example test + batch evaluation.

Covers generation exercises 1-3.

Run with:
    uv run rag_gen.py
"""

import os
from typing import Optional

import duckdb
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from tqdm import tqdm

EMB_MODEL_NAME = "qwen3-embedding-8b"
GEN_MODEL_NAME = "gemma4-26b-moe"
COLLECTION_NAME = "nace-collection"
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


class NaceClassificationResult(BaseModel):
    nace_code: Optional[str] = Field(description="Chosen NACE code from the candidate list, or null")
    codable: bool = Field(description="False if the description is too vague to code")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")


# ── RAG pipeline (shared) ─────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(override=True)

    client_llm = OpenAI(base_url=os.environ["LLMLAB_URL"], api_key=os.environ["LLMLAB_API_KEY"])
    client_qdrant = _qdrant_client_from_env()

    # ── Exercise 1: single example ────────────────────────────────────────────
    print("=== Exercise 1: Single example ===")
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
    import json
    print("\nLLM result:")
    print(json.dumps({k: v for k, v in result.items() if k != "retrieved_codes"}, indent=2))

    # ── Exercises 2 + 3: batch inference + evaluation ─────────────────────────
    print("\n=== Exercises 2 + 3: Batch inference + evaluation ===")
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
