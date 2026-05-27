import os
from functools import lru_cache

import mlflow
import numpy as np
import polars as pl
from dotenv import load_dotenv
from flask import Flask, render_template_string, request
from torchTextClassifiers import torchTextClassifiers

load_dotenv(override=True)

APP_TITLE = "Very NACE"
EXPERIMENT_NAME = "funathon-project2"
DEFAULT_RUN_ID = "3fe714a946c149b589305ec153fdc36f"
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
      textarea, input[type="number"] { border:1px solid var(--line); border-radius:12px; background:rgba(2,12,8,.7); color:#eafff4; }
      textarea { width:100%; min-height:132px; margin-top:.45rem; padding:.88rem .95rem; font:inherit; resize:vertical; }
      .controls { margin-top:.9rem; display:flex; gap:.7rem; align-items:center; flex-wrap:wrap; }
      input[type="number"] { width:90px; padding:.5rem .55rem; font:inherit; }
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
        <p>MLflow run: {{ run_id }}</p>
      </section>
      <section class="grid">
        <div class="card">
          <form id="predict-form" method="post">
            <label for="text">Business Activity Text</label>
            <textarea id="text" name="text" placeholder="e.g. Urban taxi passenger transport service">{{ text }}</textarea>
            <div class="controls">
              <label for="top_k">Top K</label>
              <input id="top_k" name="top_k" type="number" min="1" max="10" value="{{ top_k }}" />
              <button id="predict-btn" type="submit"><span class="btn-text">Run Prediction</span><span class="btn-loader" aria-hidden="true"></span></button>
            </div>
          </form>
          {% if error %}<div class="error"><strong>Error:</strong> {{ error }}</div>{% endif %}
        </div>
        {% if predictions %}
          <div class="card results-card">
            <h2 class="glitch" data-text="Predictions">Predictions</h2>
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
      if (form) form.addEventListener("submit", () => document.body.classList.add("predicting"));

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
