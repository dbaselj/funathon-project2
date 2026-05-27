# Very NACE 2.1 - Project Documentation

## Overview
This repository contains:
- A **Supervised** NACE classifier (MLflow-backed torchTextClassifiers model)
- A **RAG** NACE classifier (Qdrant retrieval + llm.lab generation)
- A unified Flask app (`app.py`) to run both modes

## Repository Structure
- `app.py`: main runtime app
- `scripts/supervised/`: supervised exercises and training scripts
- `scripts/rag/`: RAG exercises and pipeline scripts
- `solutions/`: reference exercise solution script(s)

## Requirements
- Python environment managed with `uv`
- SSPCloud services/credentials for:
  - MLflow (for supervised model artifacts)
  - Qdrant (for RAG retrieval)
  - llm.lab (for embeddings + generation)

## Setup
From repo root:

```bash
uv sync
```

Create `.env` with:

```env
# Supervised / MLflow
MLFLOW_TRACKING_URI=...
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...

# RAG / Qdrant
QDRANT_URL=https://user-<namespace>-qdrant.user.lab.sspcloud.fr/
QDRANT_API_KEY=...
QDRANT_API_PORT=443

# RAG / llm.lab
LLMLAB_URL=https://llm.lab.sspcloud.fr/api
LLMLAB_API_KEY=...
```

Optional VS Code setting already present:
- `.vscode/settings.json`: `python.terminal.useEnvFile = true`

## Run the App

```bash
uv run app.py
```

Open:
- `http://127.0.0.1:8000`

## App Behavior
### Engine: Supervised
- Loads MLflow model artifacts from configured run
- Produces a final decision and ranked alternatives

### Engine: RAG
- Embeds input text with `qwen3-embedding-8b`
- Retrieves candidates from Qdrant collection `nace-collection`
- Uses selected generation model (`gemma4-26b-moe` or `qwen3-6-35b-moe`)
- Produces final decision and alternatives
- If LLM returns null, app falls back to top retrieved code for consistent UX

## Notes on Performance
RAG is slower than Supervised because it performs remote calls:
1. embeddings API
2. Qdrant retrieval
3. chat completion generation

Current optimizations in app:
- In-memory cache for RAG calls
- Shortened retrieved context before generation

## Rebuild / Fresh Workspace Checklist
1. Clone repo and checkout desired branch
2. `uv sync`
3. Restore `.env`
4. Verify connectivity:
   - MLflow auth
   - llm.lab model list
   - Qdrant collection access
5. Run `uv run app.py`

## Exercise Scripts
### Supervised
- `scripts/supervised/exercise1_setup.py` to `exercise5_setup.py`
- `scripts/supervised/train_strong.py`
- `scripts/supervised/train_improved.py`

### RAG
- `scripts/rag/rag_vdb_ex1_connections.py`
- `scripts/rag/rag_vdb_ex2_handling_nace.py`
- `scripts/rag/rag_vdb_ex3_vector_store.py`
- `scripts/rag/rag_gen_ex1_single_example.py`
- `scripts/rag/rag_gen_ex2_ex3_eval.py`
