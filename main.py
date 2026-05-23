import json
import google.generativeai as genai
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
# production: pindahkan ke DB + scraping pipeline yang lebih rapi
app.state.fairness_acc = init_fairness_accumulator()


# ===== ModelBridge trained assets =====
SKILLBRIDGE_MODEL_PATH = "skillbridge.keras"  # (not used in runtime)
SKILLBRIDGE_SAVEDMODEL_PATH = "skillbridge_savedmodel"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_API_KEY = "AIzaSyBp3zVlMLasbyfkBF_mnGeF-nrGIEJo-4A"

genai.configure(api_key=GEMINI_API_KEY)

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

    outputs = bert_backbone(
        inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )[0]

    mask = tf.cast(tf.expand_dims(inputs["attention_mask"], -1), tf.float32)
    sum_embeddings = tf.reduce_sum(outputs * mask, axis=1)
    sum_mask = tf.clip_by_value(
        tf.reduce_sum(mask, axis=1), 1e-9, tf.float32.max
    )
    return (sum_embeddings / sum_mask).numpy()


@app.get("/")
def health_check():
    status = "Online" if bert_backbone else "Offline (Model Error)"
    return {"status": status, "engine": "SkillBridge AI"}


def _predict_cocok_tidak(cv_text: str, job_text: str):
    """Prediksi 0/1 (Tidak Cocok/Cocok) dari skillbridge_savedmodel."""
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
    """Mapping SKKNI unit via cosine similarity terhadap embedding unit pada skkni_embeddings.json."""
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


@app.post("/api/v1/match")
async def calculate_match(data: MatchRequest):
    try:
        emb_cv = get_embedding(data.cv_text)
        emb_job = get_embedding(data.job_text)
        sim = cosine_similarity(emb_cv, emb_job)[0][0]
        score = round(float(sim) * 100, 2)

        cocok_label, cocok_prob = None, None
        try:
            cocok_label, cocok_prob = _predict_cocok_tidak(
                data.cv_text, data.job_text)
        except Exception:
            cocok_label, cocok_prob = None, None

        # Privacy by design: redact sebelum dikirim ke Gemini
        safe_cv = redact_pii(data.cv_text)
        safe_job = redact_pii(data.job_text)

        # Explainable (SKKNI unit)
        top_units = _topk_skkni(data.cv_text, data.job_text, top_k=5)
        formatted_top_units = [
            {"kode_unit": u, "judul_unit": t, "similarity": round(s, 6)}
            for (u, t, s) in top_units
        ]

        # MVP gap: unit yang sim < gap_threshold
        gap_threshold = max(-0.05, (float(sim) - 0.2) / 1.0)
        gap_units = []
        for (u, t, s) in top_units:
            if s < gap_threshold:
                gap_units.append(
                    {"kode_unit": u, "judul_unit": t,
                        "similarity": round(s, 6)}
                )

        # Cache in-memory
        cache_key = hash(
            (safe_cv, safe_job, score, cocok_label, cocok_prob,
             str(formatted_top_units), str(gap_units))
        )
        if not hasattr(app.state, "gemini_cache"):
            app.state.gemini_cache = {}
        cached = app.state.gemini_cache.get(cache_key)

        # fairness proxy label (bahasa + panjang CV)
        group_label = infer_group_label_bahasa_panjang(data.cv_text)
        if cached is None:

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
                model = genai.GenerativeModel("gemini-2.0-flash")
                cached = model.generate_content(analysis_prompt).text
                app.state.gemini_cache[cache_key] = cached
            except Exception:
                # Jika quota/token Gemini habis, jangan bikin endpoint gagal.
                # Return fallback agar engine tetap bisa dipakai untuk matching.
                cached = "(Gemini quota habis) Analisis tidak tersedia saat ini."
                app.state.gemini_cache[cache_key] = cached

        # fairness audit event (proxy: bahasa + panjang CV)
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

        return {
            "success": True,
            "match_score": score,
            "prediksi_cocok": cocok_label,
            "prob_cocok": cocok_prob,
            "top_units": formatted_top_units,
            "gap_units": gap_units,
            "ai_analysis": cached,
        }

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
            {"kode_unit": u, "judul_unit": t, "similarity": round(s, 6)}
            for (u, t, s) in top_units
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

    trial_models = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-pro"]
    for model_name in trial_models:
        try:
            try:
                model = genai.GenerativeModel(model_name)
            except Exception:
                continue

            safe_cv = redact_pii(data.cv_text)
            safe_job = redact_pii(data.job_text)

            prompt = (
                "Buat 5 pertanyaan interview teknis SKKNI dalam Bahasa Indonesia. "
                f"CV: {safe_cv[:250]}. Job: {safe_job[:250]}. Unit: {data.skkni_unit}."
            )

            response = model.generate_content(prompt)
            return {"success": True, "interview_content": response.text}
        except Exception:
            continue

    raise HTTPException(status_code=500, detail="Gemini API Error")
