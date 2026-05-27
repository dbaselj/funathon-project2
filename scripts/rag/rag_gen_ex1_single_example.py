"""RAG Generation Exercise 1: single-example pipeline test."""

import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

# Models
EMB_MODEL_NAME = "qwen3-embedding-8b"
GEN_MODEL_NAME = "gemma4-26b-moe"

# Qdrant
COLLECTION_NAME = "nace-collection"
RETRIEVER_LIMIT = 5

# Generation
TEMPERATURE = 0.1

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

    activity = "Installation, maintenance and repair of residential air conditioning systems for private customers"
    print("Activity:", activity)

    # Q1: embed
    response = client_llmlab.embeddings.create(model=EMB_MODEL_NAME, input=activity)
    search_embedding = response.data[0].embedding
    print("Embedding length:", len(search_embedding))

    # Q2: retrieve
    points = client_qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=search_embedding,
        limit=RETRIEVER_LIMIT,
    )
    descriptions_retrieved = []
    codes_retrieved = []
    for point in points.model_dump()["points"]:
        descriptions_retrieved.append(point["payload"]["text"])
        codes_retrieved.append(point["payload"]["code"])

    print("Retrieved codes:", codes_retrieved)

    # Q3: generate
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
    llm_response: NaceClassificationResult = gen_response.choices[0].message.parsed
    print("\nLLM parsed result:")
    print(json.dumps(llm_response.model_dump(), indent=2))


if __name__ == "__main__":
    main()
