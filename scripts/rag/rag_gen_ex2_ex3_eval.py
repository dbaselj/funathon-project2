"""RAG Generation Exercises 2 and 3: batch coding + evaluation."""

import os
from typing import Optional

import duckdb
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from tqdm import tqdm

# Models
EMB_MODEL_NAME = "qwen3-embedding-8b"
GEN_MODEL_NAME = "gemma4-26b-moe"

# Qdrant
COLLECTION_NAME = "nace-collection"
RETRIEVER_LIMIT = 5

# Generation
TEMPERATURE = 0.1

# Evaluation
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


class NaceClassificationResult(BaseModel):
    nace_code: Optional[str] = Field(
        description="Chosen NACE code from the candidate list, or null"
    )
    codable: bool = Field(description="False if the description is too vague to code")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )


def run_rag_pipeline(
    activity: str, client_llmlab: OpenAI, client_qdrant: QdrantClient
) -> dict:
    emb_response = client_llmlab.embeddings.create(model=EMB_MODEL_NAME, input=activity)
    embedding = emb_response.data[0].embedding

    points = client_qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=RETRIEVER_LIMIT,
    )
    descriptions_retrieved = []
    codes_retrieved = []
    for point in points.model_dump()["points"]:
        descriptions_retrieved.append(point["payload"]["text"])
        codes_retrieved.append(point["payload"]["code"])

    user_prompt = USER_PROMPT_TEMPLATE.format(
        activity=activity,
        proposed_nace_descriptions="## " + "\n\n## ".join(descriptions_retrieved),
        proposed_nace_codes=", ".join(codes_retrieved),
    )

    gen_response = client_llmlab.chat.completions.parse(
        model=GEN_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        response_format=NaceClassificationResult,
    )

    result = gen_response.choices[0].message.parsed.model_dump()
    result["retrieved_codes"] = codes_retrieved
    return result


def main() -> None:
    load_dotenv(override=True)
    client_llmlab = OpenAI(
        base_url=os.environ["LLMLAB_URL"],
        api_key=os.environ["LLMLAB_API_KEY"],
    )
    client_qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        port=os.environ["QDRANT_API_PORT"],
        check_compatibility=False,
    )

    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    query_definition = f"""
    SELECT *
    FROM read_parquet(
      'https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet'
    )
    USING SAMPLE {SAMPLE_SIZE}
    """
    annotations = con.sql(query_definition).to_df().to_dict(orient="records")
    print(f"Dataset loaded: {len(annotations)} rows")

    # Exercise 2: batch inference
    records = []
    for row in tqdm(annotations, total=len(annotations), desc="Coding"):
        activity_label = row["label"]
        true_code = row["code"]
        try:
            pred = run_rag_pipeline(activity_label, client_llmlab, client_qdrant)
        except Exception as e:
            pred = {
                "nace_code": None,
                "codable": False,
                "confidence": 0.0,
                "retrieved_codes": [],
            }
            print(f"Error for '{activity_label[:60]}...': {e}")

        records.append(
            {
                "activity": activity_label,
                "true_code": true_code,
                "pred_code": pred.get("nace_code"),
                "codable": pred.get("codable", False),
                "confidence": pred.get("confidence", 0.0),
                "retrieved_codes": pred.get("retrieved_codes", []),
            }
        )

    results = pd.DataFrame(records)
    print(f"Inference complete: {len(results)} activities processed")

    # Exercise 3: evaluation metrics
    results["retriever_hit"] = results.apply(
        lambda row: row["true_code"] in row["retrieved_codes"], axis=1
    )
    results["pipeline_correct"] = results["pred_code"] == results["true_code"]
    results["llm_correct_given_retriever"] = results.apply(
        lambda row: row["pipeline_correct"] if row["retriever_hit"] else None, axis=1
    )

    retriever_accuracy = results["retriever_hit"].mean()
    llm_cond_accuracy = results.loc[
        results["retriever_hit"], "pipeline_correct"
    ].mean()
    pipeline_accuracy = results["pipeline_correct"].mean()

    print("\nMetrics:")
    print(f"Retriever@{RETRIEVER_LIMIT} accuracy: {retriever_accuracy:.1%}")
    print(f"LLM conditional accuracy: {llm_cond_accuracy:.1%}")
    print(f"Pipeline accuracy: {pipeline_accuracy:.1%}")


if __name__ == "__main__":
    main()
