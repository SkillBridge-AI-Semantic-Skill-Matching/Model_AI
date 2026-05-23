# Deployment Guide — SkillBridge AI Engine (FastAPI)

## 0) Gambaran singkat

Repo ini adalah backend API berbasis **FastAPI** yang menjalankan:

- Embedder multilingual via `transformers` (`paraphrase-multilingual-MiniLM-L12-v2`)
- Classifier via **TensorFlow SavedModel** di folder `skillbridge_savedmodel/`
- SKKNI unit scoring via `skkni_embeddings.json`
- Mock interview via **Google Gemini API**

Start command sudah disediakan di `Procfile`.

---

## 1) Persiapan sebelum deploy

### A. Pastikan semua file model ikut ter-commit

Wajib ada di root project saat deploy:

- `main.py`
- `Procfile`
- `requirements.txt`
- folder `skillbridge_savedmodel/` (isi `.pb`, `variables/`, `assets/`)
- `skkni_embeddings.json`

### B. Jangan hardcode GEMINI_API_KEY (disarankan)

Di `main.py` saat ini ada:

```py
GEMINI_API_KEY = "..."
```

Untuk produksi, ganti menjadi environment variable.
Contoh yang disarankan:

```py
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
```

---

## 2) Deploy ke Render (rekomendasi)

Render cocok dengan `Procfile` dan Python web service.

### Langkah

1. Push repo ke GitHub.
2. Buka Render → **New Web Service**.
3. Pilih:
   - Environment: **Python**
4. Konfigurasi:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - (Opsional) gunakan `Procfile` jika Render otomatis membaca.
5. Tambahkan Environment Variables:
   - `GEMINI_API_KEY=<key Anda>`
6. Deploy.

### Uji setelah deploy

- Akses: `GET /` (endpoint health check)
  - Harus balas `status: Online` jika model berhasil load.

---

## 3) Deploy ke Heroku (opsional, karena ada Procfile)

### Langkah

1. Push repo ke GitHub.
2. Buat Heroku app.
3. Pastikan `Procfile` ada (sudah ada di repo).
4. Set config var:
   - `GEMINI_API_KEY=<key Anda>`
5. Deploy.

---

## 4) Deploy ke VPS (manual)

### Langkah

1. Clone/transfer repo ke server.
2. Install dependensi:
   - `python -m venv venv && venv\Scripts\activate` (Windows VPS) / `source venv/bin/activate` (Linux)
   - `pip install -r requirements.txt`
3. Jalankan:
   - `uvicorn main:app --host 0.0.0.0 --port 8000`
4. Proxy dengan Nginx untuk HTTPS.

---

## 5) Endpoint yang tersedia

- `GET /`
- `POST /api/v1/match` body: `{ "cv_text": "...", "job_text": "..." }`
- `POST /api/v1/match-topk` body: `{ "cv_text": "...", "job_text": "...", "top_k": 5, "similarity_threshold": 0.25 }`
- `POST /api/v1/mock-interview` body: `{ "cv_text": "...", "job_text": "...", "skkni_unit": "General Professionalism" }`

---

## 6) Troubleshooting cepat

1. **Error download model transformers**
   - Pastikan server punya internet saat startup/build.
2. **Model SavedModel tidak terbaca**
   - Pastikan folder `skillbridge_savedmodel/` ikut ter-upload.
3. **Gemini API Error / 401**
   - Pastikan `GEMINI_API_KEY` benar dan diset environment variable.
4. **API lambat saat cold start**
   - First start biasanya lama karena model load. Gunakan plan yang tidak sering sleep (Render/Heroku berbayar).
