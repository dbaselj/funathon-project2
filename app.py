import os
from functools import lru_cache
from typing import Optional

import mlflow
import numpy as np
import polars as pl
import duckdb
from dotenv import load_dotenv
from flask import Flask, render_template_string, request
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from torchTextClassifiers import torchTextClassifiers

load_dotenv(override=True)

APP_TITLE = "Very NACE 2.1"
EXPERIMENT_NAME = "funathon-2026-project2"
DEFAULT_RUN_ID = "3fe714a946c149b589305ec153fdc36f"
DATA_URL = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
NACE_TSV_URL = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/NACE_Rev2.1_Structure_Explanatory_Notes_EN.tsv"

RAG_COLLECTION_NAME = "nace-collection"
RAG_EMB_MODEL_NAME = "qwen3-embedding-8b"
RAG_GEN_MODEL_NAME = "gemma4-26b-moe"
RAG_GEN_MODEL_OPTIONS = ["gemma4-26b-moe", "qwen3-6-35b-moe"]
RAG_TEMPERATURE = 0.1
DEFAULT_TOP_K = 5

SYSTEM_PROMPT = """\
You are an expert classifier for the NACE 2.1 nomenclature.
Given a company activity description and candidate NACE codes, pick the single most appropriate code from the candidates.
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
    nace_code: Optional[str] = Field(description="Chosen NACE code from the candidate list, or null")
    codable: bool = Field(description="False if the description is too vague to code")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")


HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{{ title }}</title>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
      :root { --ink:#dfffea; --ink-dim:#8fd8b6; --accent:#21f58f; --accent-2:#4ef7f9; --card:rgba(7,20,16,0.78); --line:rgba(92,255,174,0.25); }
      * { box-sizing: border-box; }
      body { margin:0; font-family:"Space Grotesk","Segoe UI",sans-serif; color:var(--ink); background:#020806; min-height:100vh; overflow-x:hidden; }
      #matrix { position:fixed; inset:0; width:100%; height:100%; z-index:0; opacity:0.24; pointer-events:none; }
      .scanlines { position:fixed; inset:0; background:repeating-linear-gradient(to bottom, rgba(255,255,255,0.03) 0px, rgba(255,255,255,0.03) 1px, transparent 2px, transparent 4px); z-index:1; pointer-events:none; animation: scanDrift 8s linear infinite; }
      .crt-flicker{position:fixed;inset:0;pointer-events:none;z-index:1;background:radial-gradient(circle at 50% 0%,rgba(120,255,200,.04),transparent 45%);mix-blend-mode:screen;animation:flicker 0.18s steps(2,end) infinite;}
      .decode-overlay{position:fixed;inset:0;display:none;z-index:3;pointer-events:none;background:linear-gradient(180deg,rgba(2,10,7,.18),rgba(2,10,7,.45));}
      body.predicting .decode-overlay{display:block;}
      .decode-overlay::before{content:"DECRYPTING NACE SIGNAL";position:absolute;left:50%;top:10%;transform:translateX(-50%);font:700 14px "IBM Plex Mono",monospace;color:#79ffd1;letter-spacing:.2em;text-shadow:0 0 8px #53ffb6;animation:blink 1s steps(2,end) infinite;}
      .shell { position:relative; z-index:2; max-width:1080px; margin:0 auto; padding:2rem 1rem; }
      .hero { background:linear-gradient(135deg, rgba(7,37,26,0.9), rgba(6,26,41,0.88)); border:1px solid var(--line); border-radius:20px; padding:1.2rem 1.35rem; box-shadow:0 16px 34px rgba(0,0,0,0.46); }
      .hero h1 { margin:0; font-size:clamp(1.55rem,4vw,2.3rem); text-shadow:0 0 10px rgba(33,245,143,0.35); }
      .hero p { margin:.45rem 0 0; font-family:"IBM Plex Mono",monospace; font-size:.86rem; color:var(--ink-dim); }
      .grid { display:grid; gap:1rem; margin-top:1rem; }
      .card { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:1rem; backdrop-filter:blur(5px); box-shadow:0 10px 26px rgba(0,0,0,.4); }
      label { font-size:.9rem; font-weight:700; color:var(--ink-dim); }
      textarea, input[type="number"], select { border:1px solid var(--line); border-radius:12px; background:rgba(2,12,8,.7); color:#eafff4; }
      textarea { width:100%; min-height:132px; margin-top:.45rem; padding:.88rem .95rem; font:inherit; resize:vertical; }
      .controls { margin-top:.9rem; display:flex; gap:.7rem; align-items:center; flex-wrap:wrap; }
      input[type="number"], select { padding:.5rem .55rem; font:inherit; }
      input[type="number"] { width:90px; }
      select { min-width:150px; }
      #predict-btn { border:1px solid rgba(78,247,249,.5); border-radius:12px; padding:.64rem 1rem; font-weight:700; color:#001108; background:linear-gradient(120deg,var(--accent),var(--accent-2)); cursor:pointer; min-width:190px; transition:transform .18s ease, box-shadow .22s ease; }
      #predict-btn:active { transform:translateY(1px) scale(0.99); }
      .btn-loader { display:none; width:16px; height:16px; border:2px solid rgba(0,17,8,.35); border-top-color:#001108; border-radius:50%; margin-left:8px; vertical-align:middle; animation:spin .85s linear infinite; }
      body.predicting #predict-btn { animation:pulseGlow .9s ease-in-out infinite; }
      body.predicting .btn-text::after { content:"..."; }
      body.predicting .btn-loader { display:inline-block; }
      .results-card { position:relative; animation:riseIn .43s cubic-bezier(.2,.8,.2,1) both; }
      .results-card::before { content:""; position:absolute; left:10px; right:10px; top:0; height:2px; background:linear-gradient(90deg,transparent,#53ffb6,#67ffff,transparent); filter:drop-shadow(0 0 6px #53ffb6); animation:sweep 2.2s linear infinite; }
      .glitch { position:relative; display:inline-block; }
      .glitch::before,.glitch::after { content:attr(data-text); position:absolute; left:0; top:0; width:100%; pointer-events:none; }
      .glitch::before { color:#7effdf; transform:translate(-1px,0); opacity:.35; }
      .glitch::after { color:#69f0ff; transform:translate(1px,0); opacity:.28; }
      table { width:100%; border-collapse:collapse; font-size:.94rem; }
      th,td { border-bottom:1px solid rgba(92,255,174,.2); padding:.62rem .45rem; text-align:left; vertical-align:top; }
      th { font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; color:#83d7ae; }
      td.code { font-family:"IBM Plex Mono",monospace; color:#b6ffe0; }
      .chip { position:relative; display:inline-block; min-width:72px; text-align:center; border-radius:999px; background:rgba(78,247,249,.15); color:#9ffcff; padding:.16rem .55rem; font-weight:700; font-family:"IBM Plex Mono",monospace; border:1px solid rgba(78,247,249,.4); overflow:hidden; }
      .chip::after{content:"";position:absolute;left:0;top:0;bottom:0;width:var(--w,0%);background:linear-gradient(90deg,rgba(33,245,143,.25),rgba(78,247,249,.35));z-index:-1;animation:barGrow .8s ease forwards;}
      .matrix-row { opacity:0; transform:translateY(8px); filter:blur(2px); }
      .matrix-row.show { animation:rowIn .42s cubic-bezier(.2,.8,.2,1) forwards; }
      .matrix-char { display:inline-block; min-width:5ch; }
      .error { margin-top:.9rem; border-radius:12px; padding:.8rem; border:1px solid #d05a5a; background:rgba(57,15,15,.55); color:#ffc3c3; }
      .mode-tag { font-size:.8rem; color:#8cdcb8; margin-left:.4rem; }
      .decision { margin:.2rem 0 .9rem; padding:.75rem .85rem; border:1px solid var(--line); border-radius:12px; }
      .decision.ok { background:rgba(33,245,143,.10); }
      .decision.warn { background:rgba(208,90,90,.16); border-color:#d05a5a; }
      .decision strong { font-family:"IBM Plex Mono",monospace; }
      @keyframes spin { to { transform:rotate(360deg);} }
      @keyframes pulseGlow { 0%{box-shadow:0 0 0 0 rgba(33,245,143,.45);}70%{box-shadow:0 0 0 16px rgba(33,245,143,0);}100%{box-shadow:0 0 0 0 rgba(33,245,143,0);} }
      @keyframes riseIn { from{opacity:0; transform:translateY(10px) scale(.994);} to{opacity:1; transform:translateY(0) scale(1);} }
      @keyframes sweep { 0%{transform:translateX(-100%);}100%{transform:translateX(100%);} }
      @keyframes rowIn { from{opacity:0; transform:translateY(8px) scale(.98); filter:blur(2px);} to{opacity:1; transform:translateY(0) scale(1); filter:blur(0);} }
      @keyframes barGrow{from{width:0}to{width:var(--w,0%)}}
      @keyframes scanDrift{0%{transform:translateY(0)}100%{transform:translateY(12px)}}
      @keyframes flicker{0%{opacity:.85}50%{opacity:.98}100%{opacity:.9}}
      @keyframes blink{50%{opacity:.45}}
    </style>
  </head>
  <body>
    <canvas id="matrix"></canvas>
    <div class="scanlines"></div>
    <div class="crt-flicker"></div>
    <div class="decode-overlay"></div>
    <div class="shell">
      <section class="hero">
        <h1 class="glitch" data-text="{{ title }}">{{ title }}</h1>
        <p>Automatic NACE Coding</p>
      </section>
      <section class="grid">
        <div class="card">
          <form id="predict-form" method="post">
            <label for="text">Business Activity Text</label>
            <textarea id="text" name="text" placeholder="e.g. Urban taxi passenger transport service">{{ text }}</textarea>
            <div class="controls">
              <label for="mode">Engine</label>
              <select id="mode" name="mode">
                <option value="rag" {% if mode == 'rag' %}selected{% endif %}>RAG</option>
                <option value="supervised" {% if mode == 'supervised' %}selected{% endif %}>Supervised</option>
              </select>
              {% if mode == 'rag' %}
              <span id="rag-model-wrap">
                <label for="rag_model">RAG Model</label>
                <select id="rag_model" name="rag_model">
                  {% for m in rag_model_options %}
                    <option value="{{ m }}" {% if m == rag_model %}selected{% endif %}>{{ m }}</option>
                  {% endfor %}
                </select>
              </span>
              {% endif %}
              <input id="top_k" name="top_k" type="hidden" value="{{ top_k }}" />
              <button id="predict-btn" type="submit"><span class="btn-text">Classify</span><span class="btn-loader" aria-hidden="true"></span></button>
            </div>
            <div class="mode-tag" style="margin-top:.55rem;">Supervised: model scores from training data. RAG: retrieve NACE context + LLM decision. Live classify: type and pause to predict automatically.</div>
          </form>
          {% if error %}<div class="error"><strong>Error:</strong> {{ error }}</div>{% endif %}
        </div>
        {% if predictions %}
          <div class="card results-card">
            <h2 class="glitch" data-text="Predictions">Predictions <span class="mode-tag">{{ mode|upper }}</span></h2>

            {% if mode == 'rag' and rag_decision %}
              <div class="decision {% if rag_decision.codable %}ok{% else %}warn{% endif %}">
                <div><strong>Final Decision:</strong> {{ rag_decision.code if rag_decision.code else 'Not codable' }}</div>
                <div>{{ rag_decision.label }}</div>
                <div>Confidence: {{ rag_decision.confidence }}</div>
              </div>
              <h3 class="mode-tag" style="margin:0 0 .35rem 0; font-size:.9rem;">Alternatives</h3>
            {% elif mode == 'supervised' and sup_decision %}
              <div class="decision ok">
                <div><strong>Final Decision:</strong> {{ sup_decision.code }}</div>
                <div>{{ sup_decision.label }}</div>
                <div>Confidence: {{ sup_decision.confidence }}</div>
              </div>
              <h3 class="mode-tag" style="margin:0 0 .35rem 0; font-size:.9rem;">Alternatives</h3>
            {% endif %}

            <table>
              <thead><tr><th>Rank</th><th>NACE Code</th><th>Label</th><th>Confidence</th></tr></thead>
              <tbody>
                {% for row in predictions %}
                  <tr class="matrix-row">
                    <td>{{ row.rank }}</td>
                    <td class="code"><span class="matrix-char" data-target="{{ row.code }}">{{ row.code }}</span></td>
                    <td>{{ row.label }}</td>
                    <td><span class="chip" data-confidence="{{ row.confidence }}"><span class="matrix-char" data-target="{{ row.confidence }}">{{ row.confidence }}</span></span></td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        {% endif %}
      </section>
    </div>
    <script>
      const form = document.getElementById("predict-form");
      const textEl = document.getElementById("text");
      const modeEl = document.getElementById("mode");
      const ragModelEl = document.getElementById("rag_model");
      const ragWrapEl = document.getElementById("rag-model-wrap");
      if (form) form.addEventListener("submit", () => document.body.classList.add("predicting"));

      let predictTimer = null;
      function schedulePredict() {
        if (!form || !textEl) return;
        const txt = (textEl.value || "").trim();
        if (txt.length < 3) return;
        clearTimeout(predictTimer);
        predictTimer = setTimeout(() => form.requestSubmit(), 550);
      }
      if (textEl) textEl.addEventListener("input", schedulePredict);
      function syncModeUI() {
        if (!modeEl || !ragWrapEl) return;
        ragWrapEl.style.display = modeEl.value === "rag" ? "inline-flex" : "none";
        ragWrapEl.style.gap = "0.45rem";
        ragWrapEl.style.alignItems = "center";
      }
      syncModeUI();
      if (modeEl) modeEl.addEventListener("change", () => { syncModeUI(); form.requestSubmit(); });
      if (ragModelEl) ragModelEl.addEventListener("change", () => form.requestSubmit());

      const canvas = document.getElementById("matrix");
      const ctx = canvas.getContext("2d");
      const chars = "01NACE$#*+~";
      let drops = [];
      let fontSize = 14;
      function resizeMatrix() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        const cols = Math.floor(canvas.width / fontSize);
        drops = Array(cols).fill(1 + Math.random() * -100);
      }
      function drawMatrix() {
        ctx.fillStyle = "rgba(2, 8, 6, 0.14)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#35f58a";
        ctx.font = fontSize + "px IBM Plex Mono";
        for (let i = 0; i < drops.length; i++) {
          const text = chars[Math.floor(Math.random() * chars.length)];
          ctx.fillText(text, i * fontSize, drops[i] * fontSize);
          if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
          drops[i]++;
        }
        requestAnimationFrame(drawMatrix);
      }
      function matrixReveal() {
        document.body.classList.add("predicting");
        setTimeout(()=>document.body.classList.remove("predicting"), 1300);
        const rows = document.querySelectorAll('.matrix-row');
        rows.forEach((row, i) => setTimeout(() => row.classList.add('show'), i * 120));
        const glyphs = "01NACE$#*+~";
        document.querySelectorAll('.chip').forEach((chip)=>{
          const c = parseFloat(chip.dataset.confidence || "0");
          chip.style.setProperty("--w", Math.max(4, Math.min(100, c*100)) + "%");
        });
        document.querySelectorAll('.matrix-char').forEach((el, idx) => {
          const target = el.dataset.target || el.textContent;
          let frame = 0;
          const total = Math.max(10, target.length * 4);
          const t = setInterval(() => {
            const out = target.split('').map((ch, i) => {
              if (ch === ' ') return ' ';
              if (i < frame / 3) return target[i];
              return glyphs[Math.floor(Math.random() * glyphs.length)];
            }).join('');
            el.textContent = out;
            frame++;
            if (frame > total) {
              el.textContent = target;
              clearInterval(t);
            }
          }, 20 + (idx % 3) * 8);
        });
      }
      window.addEventListener("resize", resizeMatrix);
      resizeMatrix();
      drawMatrix();
      matrixReveal();
      setInterval(()=>{
        const g=document.querySelectorAll(".glitch");
        g.forEach(el=>el.style.transform=`translate(${(Math.random()-.5)*0.8}px,${(Math.random()-.5)*0.8}px)`);
      },140);
    </script>
  </body>
</html>
"""

app = Flask(__name__)


@lru_cache(maxsize=1)
def load_code_labels() -> dict[str, str]:
    # Use official NACE table so hierarchical codes (e.g., 77.1, 49.1) resolve.
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    rows = con.execute(
        f"SELECT CODE, HEADING FROM read_csv('{NACE_TSV_URL}')"
    ).fetchall()
    return {
        str(code).strip(): str(heading).strip()
        for code, heading in rows
        if code and heading
    }


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
def load_supervised_model() -> tuple[torchTextClassifiers, str]:
    run_id = os.getenv("MODEL_RUN_ID", "").strip() or _latest_run_id()
    local_dir = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model_artifacts")
    model = torchTextClassifiers.load(local_dir)
    model.pytorch_model.eval()
    return model, run_id


@lru_cache(maxsize=1)
def load_rag_clients() -> tuple[OpenAI, QdrantClient]:
    llm = OpenAI(base_url=os.environ["LLMLAB_URL"], api_key=os.environ["LLMLAB_API_KEY"])
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        port=os.environ["QDRANT_API_PORT"],
        check_compatibility=False,
    )
    return llm, qdrant


def predict_supervised(text: str, top_k: int, code_labels: dict[str, str]) -> tuple[dict, list[dict]]:
    model, _run_id = load_supervised_model()
    results = model.predict(np.array([text]), top_k=top_k)
    codes = results["prediction"][0]
    confs = results["confidence"][0]
    decision = {
        "code": codes[0],
        "label": code_labels.get(codes[0], "(label not found)"),
        "confidence": f"{float(confs[0]):.3f}",
    }
    rows = [
        {
            "rank": i,
            "code": codes[i],
            "label": code_labels.get(codes[i], "(label not found)"),
            "confidence": f"{float(confs[i]):.3f}",
        }
        for i in range(1, top_k)
    ]
    return decision, rows


def predict_rag(text: str, top_k: int, code_labels: dict[str, str], rag_model: str) -> tuple[dict, list[dict]]:
    client_llm, client_qdrant = load_rag_clients()

    emb = client_llm.embeddings.create(model=RAG_EMB_MODEL_NAME, input=text).data[0].embedding
    points = client_qdrant.query_points(
        collection_name=RAG_COLLECTION_NAME,
        query=emb,
        limit=top_k,
    )

    retrieved = points.model_dump()["points"]
    codes_retrieved = [p["payload"]["code"] for p in retrieved]
    desc_retrieved = [p["payload"]["text"] for p in retrieved]

    user_prompt = USER_PROMPT_TEMPLATE.format(
        activity=text,
        proposed_nace_descriptions="## " + "\n\n## ".join(desc_retrieved),
        proposed_nace_codes=", ".join(codes_retrieved),
    )

    response = client_llm.chat.completions.parse(
        model=rag_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=RAG_TEMPERATURE,
        response_format=NaceClassificationResult,
    )
    parsed: NaceClassificationResult = response.choices[0].message.parsed

    chosen = parsed.nace_code
    decision = {
        "code": chosen,
        "label": code_labels.get(chosen, "Not codable") if chosen else "Not codable",
        "codable": bool(parsed.codable and chosen),
        "confidence": f"{float(parsed.confidence):.3f}",
    }

    rows: list[dict] = []
    rank = 1
    for p in retrieved:
        code = p["payload"]["code"]
        rows.append(
            {
                "rank": rank,
                "code": code,
                "label": code_labels.get(code, "(label not found)"),
                "confidence": f"{float(p.get('score', 0.0)):.3f}",
            }
        )
        rank += 1
        if len(rows) >= top_k:
            break
    return decision, rows


@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    predictions: list[dict] = []
    rag_decision = None
    sup_decision = None
    error = ""
    top_k = DEFAULT_TOP_K
    mode = "rag"
    rag_model = RAG_GEN_MODEL_NAME

    try:
        code_labels = load_code_labels()
        # Preload supervised model to fail fast on startup issues.
        load_supervised_model()
    except Exception as exc:
        return render_template_string(
            HTML,
            title=APP_TITLE,
            text=text,
            top_k=top_k,
            mode=mode,
            predictions=predictions,
            rag_decision=rag_decision,
            sup_decision=sup_decision,
            rag_model=rag_model,
            rag_model_options=RAG_GEN_MODEL_OPTIONS,
            error=str(exc),
        )

    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        mode = (request.form.get("mode") or "rag").strip().lower()
        if mode not in {"supervised", "rag"}:
            mode = "rag"
        rag_model = (request.form.get("rag_model") or RAG_GEN_MODEL_NAME).strip()
        if rag_model not in RAG_GEN_MODEL_OPTIONS:
            rag_model = RAG_GEN_MODEL_NAME
        try:
            top_k = int(request.form.get("top_k") or DEFAULT_TOP_K)
        except ValueError:
            top_k = DEFAULT_TOP_K
        top_k = max(1, min(10, top_k))

        if text:
            try:
                if mode == "rag":
                    rag_decision, predictions = predict_rag(text, top_k, code_labels, rag_model)
                else:
                    sup_decision, predictions = predict_supervised(text, top_k, code_labels)
            except Exception as exc:
                error = str(exc)
        else:
            error = "Please enter some text."

    return render_template_string(
        HTML,
        title=APP_TITLE,
        text=text,
        top_k=top_k,
        mode=mode,
        predictions=predictions,
        rag_decision=rag_decision,
        sup_decision=sup_decision,
        rag_model=rag_model,
        rag_model_options=RAG_GEN_MODEL_OPTIONS,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
