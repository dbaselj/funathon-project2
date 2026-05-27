import os
from functools import lru_cache

import duckdb
import mlflow
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template_string, request
from openai import OpenAI
from qdrant_client import QdrantClient
from torchTextClassifiers import torchTextClassifiers

load_dotenv(override=True)

APP_TITLE = "Very NACE 2.1"
EXPERIMENT_NAME = "funathon-2026-project2"
DEFAULT_RUN_ID = "3fe714a946c149b589305ec153fdc36f"
NACE_TSV_URL = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/NACE_Rev2.1_Structure_Explanatory_Notes_EN.tsv"

RAG_COLLECTION_NAME = "nace-collection"
RAG_EMB_MODEL_NAME = "qwen3-embedding-8b"
DEFAULT_TOP_K = 5
RAG_CACHE_SIZE = 256
REQUIRED_RAG_ENV = ("LLMLAB_URL", "LLMLAB_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")


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
      body.classifying .decode-overlay{display:block;}
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
      body.classifying #predict-btn { animation:pulseGlow .9s ease-in-out infinite; }
      body.classifying .btn-text::after { content:"..."; }
      body.classifying .btn-loader { display:inline-block; }
      .loading-status { display:none; margin-top:.85rem; color:#9fffd0; font:500 .82rem "IBM Plex Mono",monospace; letter-spacing:.02em; }
      body.classifying .loading-status { display:block; }
      .loading-track { position:relative; height:9px; margin-top:.45rem; border:1px solid rgba(78,247,249,.35); border-radius:999px; background:rgba(2,12,8,.75); overflow:hidden; }
      .loading-bar { position:absolute; inset:0 auto 0 0; width:42%; border-radius:inherit; background:linear-gradient(90deg,var(--accent),var(--accent-2)); box-shadow:0 0 14px rgba(33,245,143,.45); animation:loadingSlide 1.05s ease-in-out infinite; }
      .results-card { position:relative; animation:riseIn .43s cubic-bezier(.2,.8,.2,1) both; }
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
      @keyframes loadingSlide { 0%{transform:translateX(-110%);}50%{transform:translateX(90%);}100%{transform:translateX(240%);} }
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
              <input id="top_k" name="top_k" type="hidden" value="{{ top_k }}" />
              <button id="predict-btn" type="submit"><span class="btn-text">Classify</span><span class="btn-loader" aria-hidden="true"></span></button>
            </div>
            <div class="loading-status" role="status" aria-live="polite">
              Classifying activity...
              <div class="loading-track" aria-hidden="true"><div class="loading-bar"></div></div>
            </div>

          </form>
          {% if error %}<div class="error"><strong>Error:</strong> {{ error }}</div>{% endif %}
        </div>
        {% if predictions %}
          <div class="card results-card">
            <h2 class="glitch" data-text="Predictions">Predictions <span class="mode-tag">{{ mode|upper }}</span></h2>

            {% if mode == 'rag' and rag_decision %}
              <div class="decision ok">
                <div style="font-size:1.25rem; font-weight:700; margin-bottom:.2rem;">{{ rag_decision.code }}: {{ rag_decision.label }}</div>
                <div style="font-size:.82rem; color:var(--ink-dim);">Confidence: {{ rag_decision.confidence }}</div>
              </div>
              <h3 class="mode-tag" style="margin:0 0 .35rem 0; font-size:.9rem;">Alternatives</h3>
            {% elif mode == 'supervised' and sup_decision %}
              <div class="decision ok">
                <div style="font-size:1.25rem; font-weight:700; margin-bottom:.2rem;">{{ sup_decision.code }}: {{ sup_decision.label }}</div>
                <div style="font-size:.82rem; color:var(--ink-dim);">Confidence: {{ sup_decision.confidence }}</div>
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
      if (form) form.addEventListener("submit", () => document.body.classList.add("classifying"));

      if (modeEl && form) modeEl.addEventListener("change", () => form.requestSubmit());

      const canvas = document.getElementById("matrix");
      const ctx = canvas.getContext("2d");
      const chars = "01NACE$#*+~";
      let drops = [];
      let fontSize = 14;
      function resizeMatrix() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        const cols = Math.floor(canvas.width / fontSize);
        drops = Array.from({ length: cols }, () => 1 + Math.random() * -100);
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


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional_env_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc


def _load_httpfs(con: duckdb.DuckDBPyConnection) -> None:
    try:
        con.execute("LOAD httpfs;")
    except duckdb.Error:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")


@lru_cache(maxsize=1)
def load_code_labels() -> dict[str, str]:
    # Use official NACE table so hierarchical codes (e.g., 77.1, 49.1) resolve.
    con = duckdb.connect(database=":memory:")
    _load_httpfs(con)
    rows = con.execute(
        f"""
        SELECT CODE, HEADING
        FROM read_csv(
            '{NACE_TSV_URL}',
            delim='\t',
            header=true,
            all_varchar=true
        )
        """
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
    local_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model_artifacts"
    )
    model = torchTextClassifiers.load(local_dir)
    model.pytorch_model.eval()
    return model, run_id


@lru_cache(maxsize=1)
def load_rag_clients() -> tuple[OpenAI, QdrantClient]:
    missing = [name for name in REQUIRED_RAG_ENV if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError("Missing required RAG environment variables: " + ", ".join(missing))

    llm = OpenAI(
        base_url=_required_env("LLMLAB_URL"),
        api_key=_required_env("LLMLAB_API_KEY"),
    )
    qdrant_kwargs = {
        "url": _required_env("QDRANT_URL"),
        "api_key": _required_env("QDRANT_API_KEY"),
        "check_compatibility": False,
    }
    port = _optional_env_int("QDRANT_API_PORT")
    if port is not None:
        qdrant_kwargs["port"] = port
    qdrant = QdrantClient(**qdrant_kwargs)
    return llm, qdrant


def predict_supervised(text: str, top_k: int, code_labels: dict[str, str]) -> tuple[dict, list[dict]]:
    model, _run_id = load_supervised_model()
    results = model.predict(np.array([text]), top_k=top_k)
    codes = results["prediction"][0]
    confs = results["confidence"][0]
    if len(codes) == 0:
        raise RuntimeError("Supervised model returned no predictions")

    best_code = str(codes[0])
    decision = {
        "code": best_code,
        "label": code_labels.get(best_code, "(label not found)"),
        "confidence": f"{float(confs[0]):.3f}",
    }
    rows = [
        {
            "rank": i + 1,
            "code": str(codes[i]),
            "label": code_labels.get(str(codes[i]), "(label not found)"),
            "confidence": f"{float(confs[i]):.3f}",
        }
        for i in range(min(top_k, len(codes)))
    ]
    return decision, rows


@lru_cache(maxsize=RAG_CACHE_SIZE)
def _predict_rag_cached(text: str, top_k: int) -> tuple[dict, list[dict]]:
    client_llm, client_qdrant = load_rag_clients()

    emb = client_llm.embeddings.create(model=RAG_EMB_MODEL_NAME, input=text).data[0].embedding
    points = client_qdrant.query_points(
        collection_name=RAG_COLLECTION_NAME,
        query=emb,
        limit=top_k,
    )

    retrieved = points.model_dump()["points"]
    if not retrieved:
        return {"code": None, "label": "Not codable", "confidence": "0.000"}, []

    rows = [
        {
            "rank": i + 1,
            "code": p["payload"]["code"],
            "label": p["payload"]["code"],
            "confidence": f"{float(p.get('score', 0.0)):.3f}",
        }
        for i, p in enumerate(retrieved)
    ]
    top = retrieved[0]
    decision = {
        "code": top["payload"]["code"],
        "label": top["payload"]["code"],
        "confidence": f"{float(top.get('score', 0.0)):.3f}",
    }
    return decision, rows


def predict_rag(text: str, top_k: int, code_labels: dict[str, str]) -> tuple[dict, list[dict]]:
    decision, rows = _predict_rag_cached(text.strip(), top_k)
    decision = {
        **decision,
        "label": code_labels.get(decision["code"], "(label not found)")
        if decision["code"]
        else "Not codable",
    }
    mapped_rows = [
        {
            **r,
            "label": code_labels.get(r["code"], "(label not found)"),
        }
        for r in rows
    ]
    return decision, mapped_rows



@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    predictions: list[dict] = []
    rag_decision = None
    sup_decision = None
    error = ""
    top_k = DEFAULT_TOP_K
    mode = "rag"

    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        mode = (request.form.get("mode") or "rag").strip().lower()
        if mode not in {"supervised", "rag"}:
            mode = "rag"
        try:
            top_k = int(request.form.get("top_k") or DEFAULT_TOP_K)
        except ValueError:
            top_k = DEFAULT_TOP_K
        top_k = max(1, min(10, top_k))

        if text:
            code_labels = {}
            try:
                code_labels = load_code_labels()
            except Exception as exc:
                error = f"Could not load NACE labels: {exc}"

            try:
                if mode == "rag":
                    rag_decision, predictions = predict_rag(text, top_k, code_labels)
                else:
                    sup_decision, predictions = predict_supervised(text, top_k, code_labels)
            except Exception as exc:
                error = f"{error}; prediction failed: {exc}" if error else str(exc)
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
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
