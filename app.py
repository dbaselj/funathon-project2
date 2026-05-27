import os
from functools import lru_cache

import mlflow
import numpy as np
import polars as pl
from dotenv import load_dotenv
from flask import Flask, render_template_string, request
from torchTextClassifiers import torchTextClassifiers

load_dotenv(override=True)

APP_TITLE = "NACE Classifier"
EXPERIMENT_NAME = "funathon-2026-project2"
DEFAULT_RUN_ID = "bda5a0a0e8494cf1ba2820d11140fd6a"
DATA_URL = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"

HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{{ title }}</title>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

      :root {
        --ink: #112027;
        --paper: #f5efe5;
        --accent: #0f8b8d;
        --accent-2: #f4a261;
        --accent-3: #2a9d8f;
        --card: rgba(255, 255, 255, 0.78);
        --line: rgba(17, 32, 39, 0.16);
      }

      * { box-sizing: border-box; }

      body {
        margin: 0;
        font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(1000px 500px at -10% -20%, #f4a26155, transparent 70%),
          radial-gradient(900px 500px at 120% 0%, #2a9d8f44, transparent 65%),
          linear-gradient(160deg, #efe8dc 0%, #f8f4ec 35%, #e8f2f1 100%);
        min-height: 100vh;
      }

      .shell {
        max-width: 1080px;
        margin: 0 auto;
        padding: 2.2rem 1rem 2rem;
      }

      .hero {
        background: linear-gradient(135deg, #1d3557 0%, #0f8b8d 85%);
        color: #fff;
        border-radius: 20px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 18px 36px rgba(15, 35, 54, 0.24);
      }

      .hero h1 {
        margin: 0;
        font-size: clamp(1.5rem, 4vw, 2.15rem);
        letter-spacing: 0.01em;
      }

      .hero p {
        margin: 0.45rem 0 0;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.87rem;
        opacity: 0.88;
      }

      .grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 1rem;
        margin-top: 1rem;
      }

      .card {
        background: var(--card);
        backdrop-filter: blur(8px);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 10px 24px rgba(17, 32, 39, 0.08);
      }

      label {
        font-size: 0.92rem;
        font-weight: 700;
      }

      textarea {
        width: 100%;
        min-height: 132px;
        margin-top: 0.45rem;
        padding: 0.88rem 0.95rem;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: #ffffffd9;
        color: var(--ink);
        font: inherit;
        resize: vertical;
      }

      .controls {
        margin-top: 0.9rem;
        display: flex;
        gap: 0.7rem;
        align-items: center;
        flex-wrap: wrap;
      }

      input[type="number"] {
        width: 90px;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.5rem 0.55rem;
        font: inherit;
        background: #fff;
      }

      button {
        appearance: none;
        border: 0;
        border-radius: 12px;
        padding: 0.62rem 1rem;
        font-weight: 700;
        font-family: "Space Grotesk", sans-serif;
        color: #fff;
        background: linear-gradient(120deg, var(--accent), var(--accent-3));
        cursor: pointer;
      }

      button:hover { filter: brightness(1.05); }

      .error {
        margin-top: 0.9rem;
        border-radius: 12px;
        padding: 0.8rem;
        border: 1px solid #c33939;
        background: #fff1f1;
        color: #8b1e1e;
      }

      .results-title {
        margin: 0 0 0.7rem;
        font-size: 1.05rem;
      }

      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.94rem;
      }

      th, td {
        border-bottom: 1px solid var(--line);
        padding: 0.62rem 0.45rem;
        text-align: left;
        vertical-align: top;
      }

      th {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #3f5862;
      }

      td.code {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.9rem;
      }

      .chip {
        display: inline-block;
        min-width: 62px;
        text-align: center;
        border-radius: 999px;
        background: #0f8b8d22;
        color: #095f61;
        padding: 0.16rem 0.55rem;
        font-weight: 700;
        font-family: "IBM Plex Mono", monospace;
      }

      @media (max-width: 700px) {
        .shell { padding: 1rem 0.75rem 1.2rem; }
        .hero { border-radius: 14px; }
        .card { border-radius: 14px; }
        th:nth-child(3), td:nth-child(3) { min-width: 220px; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <section class="hero">
        <h1>{{ title }}</h1>
        <p>MLflow run: {{ run_id }}</p>
      </section>

      <section class="grid">
        <div class="card">
          <form method="post">
            <label for="text">Business Activity Text</label>
            <textarea id="text" name="text" placeholder="e.g. Urban taxi passenger transport service">{{ text }}</textarea>

            <div class="controls">
              <label for="top_k">Top K</label>
              <input id="top_k" name="top_k" type="number" min="1" max="10" value="{{ top_k }}" />
              <button type="submit">Run Prediction</button>
            </div>
          </form>

          {% if error %}
            <div class="error"><strong>Error:</strong> {{ error }}</div>
          {% endif %}
        </div>

        {% if predictions %}
          <div class="card">
            <h2 class="results-title">Predictions</h2>
            <table>
              <thead>
                <tr><th>Rank</th><th>NACE Code</th><th>Label</th><th>Confidence</th></tr>
              </thead>
              <tbody>
                {% for row in predictions %}
                  <tr>
                    <td>{{ row.rank }}</td>
                    <td class="code">{{ row.code }}</td>
                    <td>{{ row.label }}</td>
                    <td><span class="chip">{{ row.confidence }}</span></td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        {% endif %}
      </section>
    </div>
  </body>
</html>
"""

app = Flask(__name__)


def _latest_run_id() -> str:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return DEFAULT_RUN_ID
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return DEFAULT_RUN_ID
    return runs.iloc[0]["run_id"]


@lru_cache(maxsize=1)
def load_model() -> tuple[torchTextClassifiers, str]:
    run_id = os.getenv("MODEL_RUN_ID", "").strip() or _latest_run_id()
    local_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model_artifacts"
    )
    model = torchTextClassifiers.load(local_dir)
    model.pytorch_model.eval()
    return model, run_id


@lru_cache(maxsize=1)
def load_code_labels() -> dict[str, str]:
    df = pl.read_parquet(DATA_URL).select(["code", "name"]).unique("code")
    return {row[0]: row[1] for row in df.iter_rows()}


@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    predictions = []
    error = ""
    top_k = 5

    try:
        model, run_id = load_model()
        code_labels = load_code_labels()
    except Exception as exc:
        return render_template_string(
            HTML,
            title=APP_TITLE,
            run_id="(failed to load)",
            text=text,
            top_k=top_k,
            predictions=predictions,
            error=str(exc),
        )

    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        try:
            top_k = int(request.form.get("top_k") or 5)
        except ValueError:
            top_k = 5
        top_k = max(1, min(10, top_k))

        if text:
            try:
                results = model.predict(np.array([text]), top_k=top_k)
                codes = results["prediction"][0]
                confs = results["confidence"][0]
                predictions = [
                    {
                        "rank": i + 1,
                        "code": codes[i],
                        "label": code_labels.get(codes[i], "(label not found)"),
                        "confidence": f"{float(confs[i]):.3f}",
                    }
                    for i in range(top_k)
                ]
            except Exception as exc:
                error = str(exc)
        else:
            error = "Please enter some text."

    return render_template_string(
        HTML,
        title=APP_TITLE,
        run_id=run_id,
        text=text,
        top_k=top_k,
        predictions=predictions,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
