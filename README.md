# Magang-in AI — Skill Matching Engine

Platform pencocokan skill mahasiswa dengan lowongan magang teknologi di Pulau Jawa menggunakan Siamese Neural Network.

## Deskripsi

Magang-in adalah platform berbasis website yang berperan sebagai portal informasi tempat magang bagi mahasiswa bidang teknologi. Komponen AI bertanggung jawab untuk:

1. **Pencocokan Skill** — Mencocokkan tech skill user dengan lowongan magang menggunakan Siamese Neural Network
2. **CV Parsing** — Mengekstrak skill dari CV (PDF) menggunakan OCR + fuzzy matching
3. **Gap Analysis & Roadmap** — Mengidentifikasi skill yang kurang dan memberikan referensi roadmap belajar

## Arsitektur Model

```
User Skills (self-declare / CV parsing)
        │
        ▼
┌─────────────────────────┐
│  Multi-Hot Encoding     │  (80 dimensi skill vocabulary)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Siamese Neural Network │  (TensorFlow Functional API)
│                         │
│  Shared Encoder:        │
│    Dense(128) → BN → DO │
│    Dense(64)  → BN → DO │
│    Dense(32)            │
│                         │
│  Custom DistanceLayer   │
│  Classifier → Sigmoid   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Hybrid Scoring         │
│  final = 0.4*model_score│
│        + 0.6*coverage   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Ranking + Gap Analysis │
│  + Roadmap Suggestion   │
└─────────────────────────┘
```

## Hasil Evaluasi Model

| Metric | Nilai |
|--------|-------|
| Accuracy | 97.38% |
| Precision | 97.88% |
| Recall | 96.85% |
| F1-Score | 97.36% |
| MAE | 0.0377 |

## Tech Stack

- **Framework**: TensorFlow 2.20 + Keras
- **API**: FastAPI + Uvicorn
- **OCR**: PyMuPDF + PyTesseract (fallback)
- **Fuzzy Matching**: RapidFuzz
- **Deployment**: Docker + Cloudflare
- **Training Environment**: Google Colab (GPU Tesla T4)

## Struktur Folder

```
magang-in-ai/
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
│
├── app.py                    ← FastAPI application (6 endpoints)
├── inference_siamese.py      ← Inference & SkillMatcher class
├── ocr_pipeline.py           ← CV parsing (OCR + fuzzy matching)
├── train_siamese_model.py    ← Training script (GradientTape)
├── pair_generation.py        ← Data preprocessing & pair generation
│
├── model/
│   ├── siamese_model.keras   ← Trained model
│   └── model_threshold.txt   ← Optimal threshold
│
├── data/
│   ├── skill_vocabulary.csv          ← 80 skill vocabulary
│   ├── magangin_jobs_cleaned.csv     ← 201 lowongan magang lokal
│   ├── training_pairs.csv            ← 2864 training pairs
│   └── tech_internship_applications.csv  ← Dummy applicant data
│
├── docs/
│   ├── evaluation_report.txt  ← Laporan evaluasi model
│   └── training_history.csv   ← Training history per epoch
│
├── logs/
│   └── fit/                   ← TensorBoard logs
│
└── tests/
    └── testing_cv.pdf         ← Sample CV untuk testing OCR
```

## Penjelasan File

### Script Python

| File | Fungsi | Kapan Dijalankan |
|------|--------|------------------|
| `pair_generation.py` | Membuat dataset training (positive & negative pairs) dari data lowongan. Setiap pair berisi pasangan (user_skill_set, job_skill_set) dengan label cocok/tidak cocok. Menggunakan strategi: exact match, subset 70-95%, over-qualified, cross-domain negative, dan hard negative. | Sekali, saat data lowongan berubah |
| `train_siamese_model.py` | Melatih Siamese Neural Network menggunakan TensorFlow Functional API + custom training loop (tf.GradientTape). Mengimplementasikan Custom Layer, Custom Loss (Contrastive Loss), dan Custom Callback. Output: model .keras + threshold + evaluation report. | Sekali, setelah pair generation |
| `inference_siamese.py` | Class `SkillMatcher` untuk melakukan prediksi kecocokan skill user dengan lowongan. Menggunakan hybrid scoring (0.4 x model_score + 0.6 x coverage_score). Dipakai oleh FastAPI. | Diimport oleh app.py |
| `ocr_pipeline.py` | Class `CVParser` untuk mengekstrak tech skill dari CV (PDF). Strategi dual: PyMuPDF (PDF teks) + PyTesseract OCR (PDF scan/gambar). Setelah teks diekstrak, dilakukan fuzzy matching (threshold 80%) terhadap skill vocabulary. | Diimport oleh app.py |
| `app.py` | FastAPI REST API yang membungkus inference dan OCR pipeline menjadi 6 endpoint HTTP. Dijalankan dengan uvicorn. | Saat deployment (server berjalan terus) |

### Data Files

| File | Isi | Jumlah |
|------|-----|--------|
| `data/skill_vocabulary.csv` | Daftar 80 tech skill yang dikenali sistem (gabungan dari dataset lokal + roadmap.sh). Menjadi "bahasa resmi" antara frontend, model, dan OCR. | 80 skill |
| `data/magangin_jobs_cleaned.csv` | Dataset lowongan magang teknologi di Pulau Jawa. Berisi judul, perusahaan, skill requirement, role, lokasi, dan roadmap URL. | 201 lowongan |
| `data/training_pairs.csv` | Dataset training untuk Siamese Network. Berisi pasangan (user_skills, job_skills, label). Positive pairs: overlap >= 70%. Negative pairs: overlap < 30%. Grey zone (30-70%) dibuang. | 2864 pairs (balanced) |
| `data/tech_internship_applications.csv` | Data dummy pelamar magang untuk keperluan demo dan testing inference. | 300 applicants |

### Model Files

| File | Isi |
|------|-----|
| `model/siamese_model.keras` | Model Siamese Network yang sudah di-training. Format TensorFlow .keras siap produksi. Ukuran: ~140KB. |
| `model/model_threshold.txt` | Optimal threshold untuk klasifikasi binary (cocok/tidak cocok) yang ditemukan saat training melalui Custom Callback. |

### Dokumentasi

| File | Isi |
|------|-----|
| `docs/evaluation_report.txt` | Laporan evaluasi lengkap: akurasi, precision, recall, F1, MAE, confusion matrix, classification report. |
| `docs/training_history.csv` | History loss dan accuracy per epoch selama training. Bisa divisualisasikan untuk melihat learning curve. |
| `docs/Training_And_Inference_SIamese.ipynb` | Notebook Google Colab berisi dokumentasi step-by-step proses training model dan inference. Bisa dijalankan ulang di Colab untuk reproduksi hasil. |
| `logs/fit/` | TensorBoard event files untuk visualisasi training secara interaktif (loss curve, accuracy curve). |

## API Endpoints

| Method | Endpoint | Fungsi |
|--------|----------|--------|
| GET | `/api/health` | Health check |
| GET | `/api/skills` | Daftar 80 skill yang dikenali |
| POST | `/api/match` | Ranking lowongan berdasarkan skill user |
| POST | `/api/predict` | Prediksi kecocokan user vs 1 job |
| POST | `/api/normalize` | Normalisasi skill input |
| POST | `/api/extract-cv` | Extract skill dari CV (PDF) |

### Contoh Request `/api/match`

```json
POST /api/match
{
  "skills": ["react", "javascript", "typescript", "css", "html"],
  "top_n": 5
}
```

### Contoh Response

```json
{
  "user_skills_normalized": ["react", "javascript", "typescript", "css", "html"],
  "unknown_skills": [],
  "total_results": 5,
  "results": [
    {
      "rank": 1,
      "title": "Front-End Developer",
      "company": "PT XYZ",
      "final_score": 0.7736,
      "model_score": 0.9341,
      "coverage_score": 0.6667,
      "match_category": "Strong Match",
      "matched_skills": ["css", "html", "javascript", "react"],
      "missing_skills": ["tailwind", "vue"],
      "n_matched": 4,
      "n_required": 6,
      "roadmap_url": "https://roadmap.sh/frontend"
    }
  ]
}
```

## Cara Menjalankan

### Development (Lokal)

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t magangin-ai .
docker run -d -p 8000:8000 --name magangin-ai magangin-ai
```

### Swagger UI

Setelah server berjalan, buka:
```
http://localhost:8000/docs
```

## Training Ulang Model

Jika ingin re-training model:

```bash
# 1. Generate pairs (jika dataset berubah)
python pair_generation.py

# 2. Training model
python train_siamese_model.py

# 3. Test inference
python inference_siamese.py
```

## TensorBoard

```bash
tensorboard --logdir=logs/fit
```

Atau di Google Colab:
```python
%load_ext tensorboard
%tensorboard --logdir logs/fit
```

## Custom Components (Deep Learning)

| Komponen | Deskripsi |
|----------|-----------|
| **DistanceLayer** (Custom Layer) | Menghitung absolute difference antara embedding user dan job |
| **ContrastiveLoss** (Custom Loss) | Loss function khusus Siamese Network: `y*D² + (1-y)*max(margin-D, 0)²` |
| **TrainingMetricsCallback** (Custom Callback) | Monitor F1, precision, recall + early stopping + optimal threshold search |

## Scoring System

```
final_score = 0.4 × model_score + 0.6 × coverage_score

Match Categories (berdasarkan coverage):
  Strong Match  : coverage >= 60%
  Partial Match : coverage 30-59%
  Low Match     : coverage < 30%
```

## Tim

**Team Artificial Intelligence** — Coding Camp DBS 2026

## Lisensi

Project ini dibuat untuk keperluan Coding Camp DBS Foundation.
