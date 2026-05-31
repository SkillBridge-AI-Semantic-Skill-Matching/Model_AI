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


SKILL_GROUPS = {
    "backend": [
        ("Python", [r"\bpython\b"]),
        ("Django", [r"\bdjango\b"]),
        ("Flask", [r"\bflask\b"]),
        ("FastAPI", [r"\bfastapi\b"]),
        ("Laravel", [r"\blaravel\b"]),
        ("Express", [r"\bexpress\b", r"\bexpressjs\b", r"\bexpress\.js\b"]),
        ("Node.js", [r"\bnode\b", r"\bnode\.js\b", r"\bnodejs\b"]),
        ("Spring Boot", [r"\bspring\b", r"\bspring boot\b"]),
        ("NestJS", [r"\bnestjs\b", r"\bnest\.js\b"]),
        ("JWT", [r"\bjwt\b", r"\bjson web token\b"]),
        ("REST API", [r"\brest api\b", r"\brestful api\b",
         r"\brestful\b", r"\brest\b"]),
        ("GraphQL", [r"\bgraphql\b"]),
        ("gRPC", [r"\bgrpc\b"]),
        ("Microservices", [r"\bmicroservices\b", r"\bmicroservice\b"]),
    ],
    "frontend": [
        ("HTML", [r"\bhtml\b", r"\bhtml5\b"]),
        ("CSS", [r"\bcss\b", r"\bcss3\b"]),
        ("JavaScript", [r"\bjavascript\b", r"\bjs\b"]),
        ("TypeScript", [r"\btypescript\b", r"\bts\b"]),
        ("React", [r"\breact\b", r"\breact\.js\b", r"\breactjs\b"]),
        ("Vue", [r"\bvue\b", r"\bvue\.js\b", r"\bvuejs\b"]),
        ("Angular", [r"\bangular\b", r"\bangularjs\b"]),
        ("Next.js", [r"\bnext\b", r"\bnext\.js\b", r"\bnextjs\b"]),
        ("Nuxt", [r"\bnuxt\b", r"\bnuxt\.js\b"]),
        ("Svelte", [r"\bsvelte\b"]),
        ("Tailwind CSS", [r"\btailwind\b", r"\btailwindcss\b"]),
        ("Bootstrap", [r"\bbootstrap\b"]),
        ("jQuery", [r"\bjquery\b"]),
    ],
    "devops": [
        ("Docker", [r"\bdocker\b", r"\bdocker-compose\b"]),
        ("Kubernetes", [r"\bkubernetes\b", r"\bk8s\b"]),
        ("Jenkins", [r"\bjenkins\b"]),
        ("Ansible", [r"\bansible\b"]),
        ("Terraform", [r"\bterraform\b"]),
        ("CI/CD", [r"\bci/cd\b", r"\bcicd\b",
         r"\bcontinuous integration\b", r"\bcontinuous deployment\b"]),
        ("Git", [r"\bgit\b", r"\bversion control\b"]),
        ("GitHub", [r"\bgithub\b"]),
        ("GitLab", [r"\bgitlab\b"]),
        ("AWS", [r"\baws\b", r"\bamazon web services\b"]),
        ("GCP", [r"\bgcp\b", r"\bgoogle cloud\b"]),
        ("Azure", [r"\bazure\b"]),
        ("Nginx", [r"\bnginx\b"]),
        ("Apache", [r"\bapache\b"]),
        ("Linux", [r"\blinux\b"]),
        ("Bash", [r"\bbash\b", r"\bshell script\b"]),
    ],
    "database": [
        ("MySQL", [r"\bmysql\b"]),
        ("PostgreSQL", [r"\bpostgresql\b", r"\bpostgres\b"]),
        ("MongoDB", [r"\bmongodb\b", r"\bmongo\b"]),
        ("Redis", [r"\bredis\b"]),
        ("SQLite", [r"\bsqlite\b"]),
        ("Oracle", [r"\boracle\b"]),
        ("MariaDB", [r"\bmariadb\b"]),
        ("Elasticsearch", [r"\belasticsearch\b"]),
    ],
    "ml": [
        ("Machine Learning", [r"\bmachine learning\b", r"\bml\b"]),
        ("Deep Learning", [r"\bdeep learning\b", r"\bdl\b"]),
        ("TensorFlow", [r"\btensorflow\b", r"\btf\b"]),
        ("PyTorch", [r"\bpytorch\b"]),
        ("Keras", [r"\bkeras\b"]),
        ("Scikit-Learn", [r"\bscikit-learn\b", r"\bsklearn\b"]),
        ("Pandas", [r"\bpandas\b"]),
        ("NumPy", [r"\bnumpy\b"]),
        ("NLP", [r"\bnlp\b", r"\bnatural language processing\b"]),
        ("Computer Vision", [r"\bcomputer vision\b", r"\bcv\b"]),
        ("Data Warehouse", [r"\bdata warehouse\b", r"\bdwh\b"]),
        ("ETL", [r"\betl\b"]),
        ("Kafka", [r"\bkafka\b"]),
    ],
    "ui_ux": [
        ("Figma", [r"\bfigma\b"]),
        ("Adobe XD", [r"\badobe xd\b"]),
        ("Sketch", [r"\bsketch\b"]),
        ("Photoshop", [r"\bphotoshop\b"]),
        ("Illustrator", [r"\billustrator\b"]),
        ("UI/UX", [r"\bui/ux\b", r"\bui ux\b",
         r"\buser interface\b", r"\buser experience\b"]),
    ],
    "business_finance": [
        ("Accounting", [r"\baccounting\b", r"\bakuntansi\b"]),
        ("Finance", [r"\bfinance\b", r"\bkeuangan\b"]),
        ("Auditing", [r"\baudit\b", r"\bauditing\b"]),
        ("Taxation", [r"\btax\b", r"\btaxation\b", r"\bpajak\b"]),
        ("Excel", [r"\bexcel\b", r"\bspreadsheet\b"]),
        ("SAP", [r"\bsap\b"]),
        ("Accurate", [r"\baccurate\b"]),
    ],
}


def extract_skills(text: str) -> set[str]:
    import re
    text_lower = text.lower()
    found = set()
    for group_name, skills in SKILL_GROUPS.items():
        for display_name, patterns in skills:
            for pat in patterns:
                if re.search(pat, text_lower):
                    found.add(display_name)
                    break
    return found


def extract_skills_matching(cv_text: str, job_text: str, sim_score: float):
    job_skills = extract_skills(job_text)
    cv_skills = extract_skills(cv_text)

    matched_skills = sorted(list(job_skills & cv_skills))
    missing_skills = sorted(list(job_skills - cv_skills))

    # Pastikan semantic selalu dalam skala 0..1 (menghindari count vs score tercampur)
    semantic_100 = max(0.0, min(100.0, float(sim_score)))
    semantic_norm = semantic_100 / 100.0

    if not job_skills:
        # tidak ada bukti keyword dari job => pakai semantic langsung
        skill_match = int(round(min(95.0, semantic_100)))
        return skill_match, matched_skills, missing_skills

    job = len(job_skills)
    matched = len(matched_skills)
    missing = len(missing_skills)

    # Min denominator menghindari overconfident saat job cuma 1 skill
    min_denominator = 3
    denom = max(job, min_denominator)

    # Keyword evidence + gap penalty dalam skala 0..1 (bukan 0..100)
    keyword_cov = matched / denom          # coverage
    gap_ratio = missing / denom           # semakin besar missing => semakin jelek

    # Gap penalty dibuat lebih meaningful namun tetap terkurang
    # quality_keyword ~ 0..1
    gap_strength = 0.9
    keyword_quality = max(0.0, keyword_cov - (gap_ratio * gap_strength))

    # confidence keyword: saat job_skills kecil, bobot keyword turun (semantic lebih dominan)
    if job <= 1:
        keyword_weight = 0.35
    elif job <= 3:
        keyword_weight = 0.55
    else:
        keyword_weight = 0.75
    semantic_weight = 1.0 - keyword_weight

    combined_norm = (keyword_quality * keyword_weight) + \
        (semantic_norm * semantic_weight)

    # Non-linear mapping supaya skor lebih spread (low variance berkurang)
    # combined_norm dekat 0 akan turun lebih cepat, dekat 1 tetap naik.
    gamma = 0.85
    combined_norm_spread = combined_norm ** gamma

    skill_match = int(round(max(0.0, min(95.0, combined_norm_spread * 100.0))))
    return skill_match, matched_skills, missing_skills


def _calibrate_prediction(p1: float, score: float):
    # Sigmoid mapping based on score to scale probability
    scale = 1.0 / (1.0 + np.exp(-0.15 * (score - 50.0)))
    calibrated_prob = float(p1 * scale)
    calibrated_prob = max(0.0, min(1.0, calibrated_prob))
    # Threshold at 0.5 on calibrated probability
    label = 1 if (calibrated_prob >= 0.5 and score >= 45.0) else 0
    return label, calibrated_prob


def _analyze_skkni(cv_text: str, job_text: str, top_k: int = 5):
    emb_cv = get_embedding(cv_text).reshape(1, -1)
    emb_job = get_embedding(job_text).reshape(1, -1)

    with open("skkni_embeddings.json", "r", encoding="utf-8") as f:
        skkni = json.load(f)

    # 1. Deduplicate by kode_unit
    seen_units = {}
    for item in skkni:
        ku = item["kode_unit"]
        if ku not in seen_units:
            seen_units[ku] = item

    # 2. Score unique SKKNI units against job embedding to find job requirements
    job_scored = []
    for ku, item in seen_units.items():
        unit_emb = np.array(item["embedding"], dtype=np.float32).reshape(1, -1)
        job_sim = float(cosine_similarity(emb_job, unit_emb)[0, 0])
        job_scored.append((item, job_sim))

    job_scored.sort(key=lambda x: x[1], reverse=True)
    # Take top 10 relevant units for this job
    job_requirements = job_scored[:10]

    # 3. Score against CV embedding
    scored_against_cv = []
    for item, job_sim in job_requirements:
        unit_emb = np.array(item["embedding"], dtype=np.float32).reshape(1, -1)
        cv_sim = float(cosine_similarity(emb_cv, unit_emb)[0, 0])
        scored_against_cv.append(
            (item["kode_unit"], item.get("judul_unit", ""), cv_sim))

    # top_units: sorted by CV similarity descending (Strengths)
    top_units = sorted(scored_against_cv,
                       key=lambda x: x[2], reverse=True)[:top_k]

    # gap_units: sorted by CV similarity ascending (Gaps / Weaknesses)
    gap_candidates = sorted(
        scored_against_cv, key=lambda x: x[2], reverse=False)
    gap_units_raw = [x for x in gap_candidates if x[2] < 0.55][:top_k]
    if not gap_units_raw:
        gap_units_raw = gap_candidates[:2]

    # Format gap units as a list of dicts
    gap_units = [
        {"kode_unit": u, "judul_unit": t, "similarity": round(s, 6)}
        for (u, t, s) in gap_units_raw
    ]

    return top_units, gap_units


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

        if score >= 75:
            kategori = "Sangat Cocok"
        elif score >= 60:
            kategori = "Cocok"
        elif score >= 45:
            kategori = "Cukup Cocok"
        else:
            kategori = "Kurang Cocok"

        skill_match, matched_skills, missing_skills = extract_skills_matching(
            data.cv_text, data.job_text, score)

        cocok_label, cocok_prob = None, None
        try:
            cocok_label, cocok_prob = _predict_cocok_tidak(
                data.cv_text, data.job_text)
        except Exception:
            cocok_label, cocok_prob = None, None

        calibrated_label, calibrated_prob = 0, 0.0
        if cocok_prob is not None:
            calibrated_label, calibrated_prob = _calibrate_prediction(
                cocok_prob, score)

        top_units, gap_units = _analyze_skkni(
            data.cv_text, data.job_text, top_k=5)
        formatted_top_units = [
            {"kode_unit": u, "judul_unit": t, "similarity": round(s, 6)} for (u, t, s) in top_units
        ]

        group_label = infer_group_label_bahasa_panjang(data.cv_text)
        add_event(
            app.state.fairness_acc,
            {
                "group_label": group_label,
                "match_score": float(score),
                "prob_cocok": float(calibrated_prob),
                "has_gap": len(gap_units) > 0,
            },
        )

        return {
            "success": True,
            "match_score": score,
            "kategori": kategori,
            "skill_match": skill_match,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "top_units": formatted_top_units,
            "gap_units": gap_units,
            "ai_analysis": _generate_ai_analysis(
                data.cv_text,
                data.job_text,
                score,
                calibrated_label,
                calibrated_prob,
                formatted_top_units,
                gap_units,
            ),
            "debug_info": {
                "prediksi_cocok": calibrated_label,
                "prob_cocok": round(calibrated_prob, 6),
                "job_skills_count": len(extract_skills(data.job_text)),
                "cv_skills_count": len(extract_skills(data.cv_text)),
                "matched_skills_count": len(matched_skills),
                "missing_skills_count": len(missing_skills),
            }
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

            if score >= 75:
                kategori = "Sangat Cocok"
            elif score >= 60:
                kategori = "Cocok"
            elif score >= 45:
                kategori = "Cukup Cocok"
            else:
                kategori = "Kurang Cocok"

            skill_match, matched_skills, missing_skills = extract_skills_matching(
                data.cv_text, job_text, score)

            # diagnostik jumlah skill yang terdeteksi (untuk audit overconfidence)
            job_skills_count = len(extract_skills(job_text))
            cv_skills_count = len(extract_skills(data.cv_text))
            matched_skills_count = len(matched_skills)
            missing_skills_count = len(missing_skills)

            cocok_label, cocok_prob = None, None
            try:
                cocok_label, cocok_prob = _predict_cocok_tidak(
                    data.cv_text, job_text)
            except Exception:
                cocok_label, cocok_prob = None, None

            calibrated_label, calibrated_prob = 0, 0.0
            if cocok_prob is not None:
                calibrated_label, calibrated_prob = _calibrate_prediction(
                    cocok_prob, score)

            top_units, gap_units = _analyze_skkni(
                data.cv_text, job_text, top_k=data.top_k)
            formatted_top_units = [
                {"kode_unit": u, "judul_unit": t, "similarity": round(s, 6)} for (u, t, s) in top_units
            ]

            group_label = infer_group_label_bahasa_panjang(data.cv_text)
            try:
                add_event(
                    app.state.fairness_acc,
                    {
                        "group_label": group_label,
                        "match_score": float(score),
                        "prob_cocok": float(calibrated_prob),
                        "has_gap": len(gap_units) > 0,
                    },
                )
            except Exception:
                pass

            # ai_analysis akan di-generate belakangan hanya untuk top-3 (berdasarkan match_score)

            results.append(
                {
                    "job_id": job_id,
                    "job_index": idx,
                    "match_score": score,
                    "kategori": kategori,
                    "skill_match": skill_match,
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "top_units": formatted_top_units,
                    "gap_units": gap_units,
                    "debug_info": {
                        "prediksi_cocok": calibrated_label,
                        "prob_cocok": round(calibrated_prob, 6),
                        "job_skills_count": job_skills_count,
                        "cv_skills_count": cv_skills_count,
                        "matched_skills_count": matched_skills_count,
                        "missing_skills_count": missing_skills_count,
                    },
                    "_llm_payload": {
                        "cv_text": data.cv_text,
                        "job_text": job_text,
                        "score": score,
                        "cocok_label": calibrated_label,
                        "cocok_prob": calibrated_prob,
                        "formatted_top_units": formatted_top_units,
                        "gap_units": gap_units,
                    },
                    "ai_analysis": None,
                }
            )

        # Sort by match_score descending, then generate AI only for top-3
        MAX_OPENROUTER_JOBS = 3
        results_sorted = sorted(
            results, key=lambda x: x["match_score"], reverse=True)
        top_results = results_sorted[:MAX_OPENROUTER_JOBS]

        for r in top_results:
            p = r["_llm_payload"]
            r["ai_analysis"] = _generate_ai_analysis(
                p["cv_text"],
                p["job_text"],
                p["score"],
                p["cocok_label"],
                p["cocok_prob"],
                p["formatted_top_units"],
                p["gap_units"],
            )

        # cleanup payload for all results
        for r in results_sorted:
            if "_llm_payload" in r:
                del r["_llm_payload"]

        return {"success": True, "results": results_sorted}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)
                            )@app.post("/api/v1/match-topk")


async def match_topk(data: MatchTopKRequest):
    try:
        sim = cosine_similarity(get_embedding(
            data.cv_text), get_embedding(data.job_text))[0][0]
        score = round(float(sim) * 100, 2)
        if score >= 75:
            kategori = "Sangat Cocok"
        elif score >= 60:
            kategori = "Cocok"
        elif score >= 45:
            kategori = "Cukup Cocok"
        else:
            kategori = "Kurang Cocok"

        skill_match, matched_skills, missing_skills = extract_skills_matching(
            data.cv_text, data.job_text, score)

        cocok_label, cocok_prob = _predict_cocok_tidak(
            data.cv_text, data.job_text)
        status = "Cocok" if cocok_label == 1 else "Tidak Cocok"
        is_trust = cocok_prob is not None and cocok_prob >= (
            1.0 - data.similarity_threshold)

        calibrated_label, calibrated_prob = 0, 0.0
        if cocok_prob is not None:
            calibrated_label, calibrated_prob = _calibrate_prediction(
                cocok_prob, score)

        top_units, _ = _analyze_skkni(
            data.cv_text, data.job_text, top_k=data.top_k)
        formatted_units = [
            {"kode_unit": u, "judul_unit": t, "similarity": round(s, 6)} for (u, t, s) in top_units
        ]

        return {
            "success": True,
            "match_score": score,
            "kategori": kategori,
            "skill_match": skill_match,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "status": status,
            "auto_assign": is_trust,
            "top_units": formatted_units,
            "debug_info": {
                "prediksi_cocok": calibrated_label,
                "prob_cocok": round(calibrated_prob, 6)
            }
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
