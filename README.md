# PELET (Pencari Lowongan Efektif & Tepat): AI Semantic Matching Berbasis SKKNI

**PELET** adalah sebuah sistem cerdas (AI) yang dibangun untuk menyelesaikan permasalahan ketidakcocokan keterampilan (*skill mismatch*) antara kompetensi pencari kerja dan kebutuhan industri. Model ini dirancang untuk memetakan *resume* / CV kandidat secara objektif ke Standar Kompetensi Kerja Nasional Indonesia (SKKNI) menggunakan arsitektur Semantic Matching.

---

## 🎯 Fitur Utama Model

1. **Semantic Matching CV ke Lowongan (SKKNI)**
   Model ini tidak lagi menggunakan teknik konvensional pencocokan kata kunci (*keyword matching*), melainkan memanfaatkan pemahaman makna bahasa (semantik) dari **Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`)**. Jika seorang kandidat menulis "ReactJS" dan lowongan meminta "Front-End Developer", sistem tetap dapat menemukan relasi kecocokannya.
2. **SKKNI Mapping & Skill Gap Analysis**
   Sistem mengekstraksi kemampuan pelamar dan memetakannya ke dalam 5 rumpun pekerjaan utama berdasarkan **SKKNI** (Programmer, UI/UX Designer, Data Analyst, Cyber Security, Keuangan).
3. **Peringkat Objektif dengan AI**
   Dilengkapi dengan *Custom Training Loop* TensorFlow untuk menghasilkan klasifikasi tingkat kecocokan yang sangat akurat.

---

## 📊 Evaluasi & Performa Model

Model *Deep Learning* ini telah dilatih dan divalidasi menggunakan lebih dari 2.500 pasang dataset industri nyata. Evaluasi akhir pada 1.175 data uji menunjukkan performa yang **sangat tinggi**, yaitu:

- **Test Accuracy**: 99,06%
- **Test MAE**: 0,0173
- **Optimal Threshold (Youden's J)**: 0,4220 (42,20%)

**Detail Classification Report (Macro Avg: 0,99):**
| Class / Label | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Tidak Cocok (0)** | 1.00 | 0.99 | 0.99 | 834 |
| **Cocok (1)** | 0.98 | 0.99 | 0.98 | 341 |

---

## ⚙️ Arsitektur & Teknologi (Tech Stack)

- **Machine Learning & NLP**: TensorFlow (`tf.keras`), HuggingFace Transformers (Sentence-BERT), Scikit-Learn
- **API Framework**: FastAPI & Uvicorn
- **Utilities**: NumPy, Pydantic
- **Model Format**: `.keras` / `SavedModel`

---

## 🚀 Panduan Instalasi & Menjalankan Model Secara Lokal

### 1. Persiapan Environment

Sangat direkomendasikan untuk menggunakan *virtual environment* Python (versi 3.8 - 3.11):
```bash
# Membuat virtual environment
python -m venv venv

# Aktivasi virtual environment (Windows)
.\venv\Scripts\activate

# Aktivasi virtual environment (Mac/Linux)
source venv/bin/activate
```

### 2. Instalasi Dependensi

Instal semua pustaka yang dibutuhkan melalui `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Struktur File Penting

- `main.py`: Titik masuk utama (API Server) FastAPI yang melayani *endpoint* model inferensi.
- `skillbridge.keras`: Bobot (*weights*) dan arsitektur model *Deep Learning* utama yang telah dilatih.
- `skillbridege.ipynb`: Notebook eksperimen, *data preparation*, dan *training* model dari awal hingga evaluasi.
- `requirements.txt`: Daftar pustaka instalasi Python.

### 4. Menjalankan Server API

Untuk melayani model dan mencoba inference melalui REST API, jalankan FastAPI menggunakan Uvicorn:
```bash
uvicorn main:app --reload
```
Secara default, *server* akan berjalan di `http://127.0.0.1:8000`. 
Anda bisa membuka dokumentasi interaktif API (Swagger UI) dengan mengakses: **`http://127.0.0.1:8000/docs`**.

---

## 📈 Pengembangan Lanjutan

Model AI saat ini dirancang sebagai *Minimum Viable Product (MVP)* untuk 5 rumpun spesifik guna memastikan akurasi sangat tinggi dan minim halusinasi.
Langkah ke depan (*Next Steps*):
- Ekstensi ke rumpun pekerjaan lain (contoh: *Digital Marketing*, *Product Management*, *Sales*).
- Implementasi API Generative AI (*Qwen / Gemini*) untuk **AI Interview Generator** dan **Skill Gap Analysis** dengan fitur proteksi PII (*Personally Identifiable Information*).
