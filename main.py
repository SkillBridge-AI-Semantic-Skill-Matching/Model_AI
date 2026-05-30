import json
import os
import requests
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, TFAutoModel
import transformers.utils.import_utils

from fairness_utils import (
    infer_group_label_bahasa_panjang,
    init_fairness_accumulator,
    add_event,
    compute_fairness_report,
)

transformers.utils.import_utils.is_torch_available = lambda: False


def redact_pii(text: str) -> str:
    """Redact PII-like patterns before sending content to external LLMs."""
    if text is None:
        return ""

    import re

    patterns = [
        (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]"),
        (r"\b\+?62[\s-]?(?:\d[\s-]?){9,11}\b", "[PHONE]"),
        (r"\b0\d{9,11}\b", "[PHONE]"),
        (r"\b\d{16}\b", "[ID_NUMBER]"),
        (r"\b\d{12}\b", "[ID_NUMBER]"),
        (r"https?://\S+|www\.\S+", "[URL]"),
    ]

    redacted = text
    for pat, repl in patterns:
        redacted = re.sub(pat, repl, redacted)
    return redacted


app = FastAPI(
    title="SkillBridge AI Engine",
    description="API for Semantic Matching and SKKNI-based Analysis",
    version="1.1.2",
)

# in-memory accumulator untuk audit fairness (MVP)
app.state.fairness_acc = init_fairness_accumulator()

# ===== ModelBridge trained assets =====
SKILLBRIDGE_SAVEDMODEL_PATH = "skillbridge_savedmodel"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ---- OpenRouter ----
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    # fallback baca file .env sederhana tanpa python-dotenv
    # (opsi B: pastikan user menaruh key di file .env)
    try:
        from pathlib import Path
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k == "OPENROUTER_API_KEY" and v:
                    OPENROUTER_API_KEY = v
                    os.environ["OPENROUTER_API_KEY"] = v
                    break
    except Exception:
        pass


OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

# Default model (bisa kamu ganti dengan yang lain dari GET /api/v1/models)
OPENROUTER_MODEL_ID = os.getenv("OPENROUTER_MODEL_ID", "qwen/qwen3.7-max")


# ---- load embedding backbone ----
tokenizer = None
bert_backbone = None
print("--- Initializing SkillBridge AI Engine ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert_backbone = TFAutoModel.from_pretrained(MODEL_NAME)
    print("[OK] NLP Models Loaded Successfully")
except Exception as e:
    print(f"[ERROR] ERROR LOADING MODELS: {e}")
    print("TIP: Pastikan internet aktif.")

# ---- load classifier (SavedModel) ----
saved_model = None
saved_sig = None
try:
    saved_model = tf.saved_model.load(SKILLBRIDGE_SAVEDMODEL_PATH)
    saved_sig = saved_model.signatures["serving_default"]
    print("[OK] Loaded skillbridge_savedmodel (SavedModel signature)")
except Exception as e:
    saved_model = None
    saved_sig = None
    print(f"[ERROR] ERROR LOADING skillbridge_savedmodel: {e}")


class MatchRequest(BaseModel):
    cv_text: str
    job_text: str


class MatchMultiRequest(BaseModel):
    cv_text: str
    jobs: list[dict]
    top_k: int = 5


class InterviewRequest(BaseModel):
    cv_text: str
    job_text: str
    skkni_unit: str = "General Professionalism"


class MatchTopKRequest(BaseModel):
    cv_text: str
    job_text: str
    top_k: int = 5
    similarity_threshold: float = 0.25


def get_embedding(text: str) -> np.ndarray:
    if tokenizer is None or bert_backbone is None:
        raise Exception(
            "Model AI belum dimuat dengan benar. Cek terminal server.")

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="tf",
    )

    outputs = bert_backbone(inputs["input_ids"],
                            attention_mask=inputs["attention_mask"])[0]

    mask = tf.cast(tf.expand_dims(inputs["attention_mask"], -1), tf.float32)
    sum_embeddings = tf.reduce_sum(outputs * mask, axis=1)
    sum_mask = tf.clip_by_value(tf.reduce_sum(
        mask, axis=1), 1e-9, tf.float32.max)
    return (sum_embeddings / sum_mask).numpy()


def _openrouter_chat(prompt: str, *, model_id: str | None = None) -> str:
    """Call OpenRouter chat/completions and return assistant text."""
    if not OPENROUTER_API_KEY:
        raise Exception("OPENROUTER_API_KEY belum di-set")

    model_id = model_id or OPENROUTER_MODEL_ID

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }

    r = requests.post(OPENROUTER_CHAT_URL, headers=headers,
                      json=payload, timeout=60)
    if r.status_code >= 400:
        # tampilkan sebagian body supaya bisa diketahui penyebabnya (quota/invalid model/rate limit/etc)
        snippet = (r.text or "")[:1200]
        raise Exception(
            f"OpenRouter error {r.status_code} (model={model_id}): {snippet}")

    data = r.json()
    # OpenAI-compatible format
    return data["choices"][0]["message"]["content"]


@app.get("/")
def health_check():
    status = "Online" if bert_backbone else "Offline (Model Error)"
    return {"status": status, "engine": "SkillBridge AI"}


def _predict_cocok_tidak(cv_text: str, job_text: str):
    if saved_sig is None:
        raise Exception(
            "skillbridge_savedmodel belum ter-load. Pastikan folder 'skillbridge_savedmodel' ada di root project."
        )

    cv_emb = get_embedding(cv_text)
    job_emb = get_embedding(job_text)

    sig_inputs = {
        "CV_Embedding_Input": tf.convert_to_tensor(cv_emb, dtype=tf.float32),
        "Job_Embedding_Input": tf.convert_to_tensor(job_emb, dtype=tf.float32),
    }

    out = saved_sig(**sig_inputs)
    pred = out["Match_Probability"].numpy()

    pred_arr = np.array(pred)
    p1 = float(pred_arr.reshape(-1)[0])
    label = 1 if p1 >= 0.5 else 0
    return label, p1


def _topk_skkni(cv_text: str, job_text: str, top_k: int = 5):
    emb_cv = get_embedding(cv_text)
    emb_job = get_embedding(job_text)
    emb_pair = (emb_cv + emb_job) / 2.0
    emb_pair = emb_pair.reshape(1, -1)

    with open("skkni_embeddings.json", "r", encoding="utf-8") as f:
        skkni = json.load(f)

    scored = []
    for item in skkni:
        unit_emb = np.array(item["embedding"], dtype=np.float32).reshape(1, -1)
        sim = float(cosine_similarity(emb_pair, unit_emb)[0, 0])
        scored.append((item["kode_unit"], item.get("judul_unit", ""), sim))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[: max(1, top_k)]


def _compute_gap_units(top_units, sim, gap_threshold):
    gap_units = []
    for u, t, s in top_units:
        if s < gap_threshold:
            gap_units.append(
                {"kode_unit": u, "judul_unit": t, "similarity": round(s, 6)})
    return gap_units


def _generate_ai_analysis(
    cv_text: str,
    job_text: str,
    score: float,
    cocok_label,
    cocok_prob,
    formatted_top_units,
    gap_units,
):

    safe_cv = redact_pii(cv_text)
    safe_job = redact_pii(job_text)

    skkni_context = json.dumps(
        {
            "top_units": formatted_top_units,
            "gap_units": gap_units,
            "match_score": score,
            "prediksi_cocok": cocok_label,
            "prob_cocok": cocok_prob,
        },
        ensure_ascii=False,
    )

    prob_str = f"{cocok_prob:.3f}" if cocok_prob is not None else "N/A"

    analysis_prompt = (
        "Kamu adalah AI Career Coach yang memberi analisis berdasarkan unit SKKNI. "
        "Wajib:\n"
        "1) jelaskan singkat kenapa skill kandidat cocok/kurang, rujuk ke top_units dan gap_units (kode_unit).\n"
        "2) berikan rekomendasi tindakan belajar/pelatihan mengacu gap_units (kode_unit).\n"
        "3) Jangan menebak data pribadi kandidat.\n\n"
        f"Data terstruktur (JSON)={skkni_context}\n"
        f"Prediksi Cocok/Tidak Cocok={cocok_label} (prob Cocok={prob_str})\n"
        f"Cuplikan CV (PII sudah disensor)={safe_cv[:250]}.\n"
        f"Cuplikan Job (PII sudah disensor)={safe_job[:250]}."
    )

    try:
        return _openrouter_chat(analysis_prompt)
    except Exception:
        # fallback: jangan tampilkan error mentah/traceback
        top_preview = formatted_top_units[:3] if formatted_top_units else []
        gap_preview = gap_units[:3] if gap_units else []
        return (
            "Analisis sementara tidak tersedia karena layanan LLM sedang bermasalah. "
            "Berikut ringkasan berbasis data SKKNI (perkiraan):\n\n"
            "1) Unit teratas (top_units) (cuplikan): "
            + ", ".join([f"{x['kode_unit']}" for x in top_preview])
            + "\n"
            "2) Unit gap (gap_units) (cuplikan): "
            + (", ".join([f"{x['kode_unit']}: {x['similarity']:.3f}" for x in gap_preview])
               if gap_preview else "Tidak terdeteksi")
            + "\n"
            "3) Rekomendasi: fokus pada gap_units untuk meningkatkan kecocokan terhadap job."
        )


@app.post("/api/v1/match")
async def calculate_match(data: MatchRequest):
    try:
        sim = cosine_similarity(get_embedding(
            data.cv_text), get_embedding(data.job_text))[0][0]
        score = round(float(sim) * 100, 2)

        cocok_label, cocok_prob = None, None
        try:
            cocok_label, cocok_prob = _predict_cocok_tidak(
                data.cv_text, data.job_text)
        except Exception:
            cocok_label, cocok_prob = None, None

        top_units = _topk_skkni(data.cv_text, data.job_text, top_k=5)
        formatted_top_units = [
            {"kode_unit": u, "judul_unit": t, "similarity": round(s, 6)} for (u, t, s) in top_units
        ]

        gap_threshold = max(-0.05, (float(sim) - 0.2) / 1.0)
        gap_units = _compute_gap_units(top_units, float(sim), gap_threshold)

        group_label = infer_group_label_bahasa_panjang(data.cv_text)
        add_event(
            app.state.fairness_acc,
            {
                "group_label": group_label,
                "match_score": float(score),
                "prob_cocok": float(cocok_prob) if cocok_prob is not None else None,
                "has_gap": len(gap_units) > 0,
            },
        )

        return {
            "success": True,
            "match_score": score,
            "prediksi_cocok": cocok_label,
            "prob_cocok": cocok_prob,
            "top_units": formatted_top_units,
            "gap_units": gap_units,
            "ai_analysis": _generate_ai_analysis(
                data.cv_text,
                data.job_text,
                score,
                cocok_label,
                cocok_prob,
                formatted_top_units,
                gap_units,
            ),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/match-multi")
async def match_multi(data: MatchMultiRequest):
    try:
        if not isinstance(data.jobs, list) or len(data.jobs) == 0:
            raise HTTPException(
                status_code=400, detail="jobs must be a non-empty array")

        results = []
        for idx, job in enumerate(data.jobs):
            if not isinstance(job, dict) or "job_text" not in job:
                raise HTTPException(
                    status_code=400,
                    detail=f"jobs[{idx}] must be an object with at least 'job_text'",
                )

            job_text = job["job_text"]
            job_id = job.get("job_id")

            emb_cv = get_embedding(data.cv_text)
            emb_job = get_embedding(job_text)
            sim = cosine_similarity(emb_cv, emb_job)[0][0]
            score = round(float(sim) * 100, 2)

            cocok_label, cocok_prob = None, None
            try:
                cocok_label, cocok_prob = _predict_cocok_tidak(
                    data.cv_text, job_text)
            except Exception:
                cocok_label, cocok_prob = None, None

            top_units = _topk_skkni(data.cv_text, job_text, top_k=data.top_k)
            formatted_top_units = [
                {"kode_unit": u, "judul_unit": t, "similarity": round(s, 6)} for (u, t, s) in top_units
            ]

            gap_threshold = max(-0.05, (float(sim) - 0.2) / 1.0)
            gap_units = _compute_gap_units(
                top_units, float(sim), gap_threshold)

            group_label = infer_group_label_bahasa_panjang(data.cv_text)
            try:
                add_event(
                    app.state.fairness_acc,
                    {
                        "group_label": group_label,
                        "match_score": float(score),
                        "prob_cocok": float(cocok_prob) if cocok_prob is not None else None,
                        "has_gap": len(gap_units) > 0,
                    },
                )
            except Exception:
                pass

            # Limit pemanggilan OpenRouter untuk mencegah quota limit
            MAX_OPENROUTER_JOBS = 3
            ai_text = _generate_ai_analysis(
                data.cv_text,
                job_text,
                score,
                cocok_label,
                cocok_prob,
                formatted_top_units,
                gap_units,
            ) if idx < MAX_OPENROUTER_JOBS else None

            results.append(
                {
                    "job_id": job_id,
                    "job_index": idx,
                    "match_score": score,
                    "prediksi_cocok": cocok_label,
                    "prob_cocok": cocok_prob,
                    "top_units": formatted_top_units,
                    "gap_units": gap_units,
                    "ai_analysis": ai_text,
                }
            )

        return {"success": True, "results": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/match-topk")
async def match_topk(data: MatchTopKRequest):
    try:
        cocok_label, cocok_prob = _predict_cocok_tidak(
            data.cv_text, data.job_text)
        status = "Cocok" if cocok_label == 1 else "Tidak Cocok"
        is_trust = cocok_prob is not None and cocok_prob >= (
            1.0 - data.similarity_threshold)

        top_units = _topk_skkni(data.cv_text, data.job_text, top_k=data.top_k)
        formatted_units = [
            {"kode_unit": u, "judul_unit": t, "similarity": round(s, 6)} for (u, t, s) in top_units
        ]

        return {
            "success": True,
            "prediksi_cocok": cocok_label,
            "prob_cocok": float(cocok_prob) if cocok_prob is not None else None,
            "status": status,
            "auto_assign": is_trust,
            "top_units": formatted_units,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/fairness-report")
def fairness_report():
    try:
        return compute_fairness_report(app.state.fairness_acc)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/v1/mock-interview")
async def generate_interview(data: InterviewRequest):
    trial_models = [
        OPENROUTER_MODEL_ID,
        "~openai/gpt-mini-latest",
        "~anthropic/claude-haiku-latest",
    ]

    last_error: str = ""  # untuk mengembalikan error paling akhir ke client
    for model_id in trial_models:
        try:
            safe_cv = redact_pii(data.cv_text)
            safe_job = redact_pii(data.job_text)

            prompt = (
                "Buat 1 pertanyaan interview teknis utama SKKNI dalam Bahasa Indonesia untuk unit: "
                f"{data.skkni_unit}. "
                f"CV (PII tersensor): {safe_cv[:200]}. Job (PII tersensor): {safe_job[:200]}. "
                "Setelah pertanyaan utama, buat 2 pertanyaan follow-up untuk menguji kedalaman kandidat. "
                "Jawab dalam format:\n"
                "- Pertanyaan Utama: ...\n"
                "- Follow-up 1: ...\n"
                "- Follow-up 2: ..."
            )

            text = _openrouter_chat(prompt, model_id=model_id)
            return {"success": True, "interview_content": text}
        except Exception as e:
            last_error = str(e)
            continue

    raise HTTPException(
        status_code=500, detail=last_error or "OpenRouter API Error")
