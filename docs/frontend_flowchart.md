# Frontend Flowchart — Integrasi API Magang-in AI

Dokumen ini menjelaskan flow frontend dan bagaimana setiap halaman berinteraksi dengan API backend.

## Base URL

```
Production: https://[subdomain].[domain].com
Local Dev:  http://192.168.30.3:8001
Swagger UI: [base_url]/docs
```

---

## Flow Utama User

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER MASUK WEBSITE                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HALAMAN ONBOARDING                           │
│                                                                   │
│  "Bagaimana kamu ingin memasukkan skill?"                        │
│                                                                   │
│  ┌─────────────────┐          ┌─────────────────┐               │
│  │  Self-Declaration │          │  Upload CV (PDF) │               │
│  │  (Pilih Manual)   │          │                  │               │
│  └────────┬──────────┘          └────────┬─────────┘               │
└───────────┼──────────────────────────────┼──────────────────────┘
            │                              │
            ▼                              ▼
┌───────────────────────┐    ┌────────────────────────────┐
│  HALAMAN SELF-DECLARE  │    │  HALAMAN UPLOAD CV          │
│                         │    │                             │
│  GET /api/skills        │    │  User upload file PDF       │
│  → Tampilkan dropdown/  │    │                             │
│    checklist 80 skill   │    │  POST /api/extract-cv       │
│                         │    │  → Kirim file PDF           │
│  User pilih skill       │    │  → Dapat extracted_skills   │
│  dari daftar            │    │                             │
│                         │    │  Tampilkan hasil:           │
│                         │    │  "Skill terdeteksi dari CV: │
│                         │    │   react, javascript, ..."   │
│                         │    │                             │
│                         │    │  User bisa edit/tambah/hapus│
│                         │    │  skill sebelum submit       │
└───────────┬─────────────┘    └──────────────┬──────────────┘
            │                                  │
            │      user_skills = [...]          │
            └──────────────┬───────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HALAMAN HASIL MATCHING                         │
│                                                                   │
│  POST /api/match                                                 │
│  Body: { "skills": user_skills, "top_n": 10 }                   │
│  → Dapat ranking lowongan                                        │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ #1 [STRONG]  Frontend Intern @ PT XYZ     │ 77% match   │    │
│  │     Matched: react, javascript, css       │             │    │
│  │     Missing: typescript, nextjs           │ [Detail →]  │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ #2 [PARTIAL] Web Developer @ PT ABC       │ 55% match   │    │
│  │     Matched: javascript, css              │             │    │
│  │     Missing: php, laravel, mysql          │ [Detail →]  │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ #3 [LOW]     Backend Dev @ PT DEF         │ 25% match   │    │
│  │     Matched: javascript                   │             │    │
│  │     Missing: node, express, sql, docker   │ [Detail →]  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  [← Ubah Skill]                                                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │ User klik [Detail →] pada 1 job
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HALAMAN DETAIL LOWONGAN                        │
│                                                                   │
│  (Opsional) POST /api/predict                                    │
│  Body: { "user_skills": [...], "job_skills": [...] }             │
│  → Atau ambil data dari response /api/match sebelumnya           │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Frontend Intern @ PT XYZ                                │    │
│  │  Location: Jakarta | Role: Frontend                      │    │
│  │                                                          │    │
│  │  Match Score: 77% (Strong Match)                         │    │
│  │  Model Score: 0.93 | Coverage: 0.67                      │    │
│  │                                                          │    │
│  │  ✓ Skill yang kamu kuasai (4/6):                         │    │
│  │    react, javascript, css, html                          │    │
│  │                                                          │    │
│  │  ✗ Skill yang perlu dipelajari (2):                      │    │
│  │    typescript, nextjs                                    │    │
│  │                                                          │    │
│  │  [Lihat Roadmap →] (link ke roadmap.sh/frontend)         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  [← Kembali ke Hasil]    [Apply Lowongan →]                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Flow Normalisasi (Background / Autocomplete)

```
┌─────────────────────────────────────────────────────────────────┐
│  SAAT USER KETIK DI INPUT FIELD (autocomplete)                   │
│                                                                   │
│  User ketik: "React.js"                                          │
│                                                                   │
│  POST /api/normalize                                             │
│  Body: { "skills": ["React.js"] }                                │
│  Response: { "normalized": ["react"], "unknown": [] }            │
│                                                                   │
│  → Tampilkan suggestion: "react" ✓                               │
│  → Kalau unknown: tampilkan warning "Skill tidak dikenali"       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Mapping Endpoint ke Halaman

| Halaman | Endpoint | Kapan Dipanggil |
|---------|----------|-----------------|
| Onboarding / Self-Declare | `GET /api/skills` | Saat halaman load (ambil daftar skill untuk dropdown) |
| Upload CV | `POST /api/extract-cv` | Saat user upload PDF |
| Hasil Matching | `POST /api/match` | Saat user submit skill (dari self-declare atau CV) |
| Detail Lowongan | `POST /api/predict` (opsional) | Saat user klik 1 lowongan |
| Input Field (autocomplete) | `POST /api/normalize` (opsional) | Saat user ketik skill manual |
| Semua halaman | `GET /api/health` | Untuk cek apakah API hidup (loading state) |

---

## Sequence Diagram

```
User          Frontend              API (FastAPI)
 │               │                       │
 │  Buka app     │                       │
 │──────────────>│                       │
 │               │  GET /api/health      │
 │               │──────────────────────>│
 │               │  {"status": "ok"}     │
 │               │<──────────────────────│
 │               │                       │
 │               │  GET /api/skills      │
 │               │──────────────────────>│
 │               │  {"skills": [...]}    │
 │               │<──────────────────────│
 │               │                       │
 │  Pilih skill  │                       │
 │  atau upload  │                       │
 │  CV           │                       │
 │──────────────>│                       │
 │               │                       │
 │  [Jika CV]    │  POST /api/extract-cv │
 │               │──────────────────────>│
 │               │  {"extracted_skills"} │
 │               │<──────────────────────│
 │               │                       │
 │  Submit       │                       │
 │──────────────>│                       │
 │               │  POST /api/match      │
 │               │──────────────────────>│
 │               │  {"results": [...]}   │
 │               │<──────────────────────│
 │               │                       │
 │  Lihat hasil  │                       │
 │<──────────────│                       │
 │               │                       │
 │  Klik detail  │                       │
 │──────────────>│                       │
 │               │  POST /api/predict    │
 │               │  (opsional)           │
 │               │──────────────────────>│
 │               │  {"final_score":...}  │
 │               │<──────────────────────│
 │               │                       │
 │  Lihat detail │                       │
 │  + roadmap    │                       │
 │<──────────────│                       │
```

---

## Detail Request & Response Per Endpoint

### 1. GET /api/health

Cek apakah API hidup dan model sudah loaded.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "vocab_size": 80,
  "jobs_count": 201,
  "scoring_formula": "0.4*model + 0.6*coverage",
  "match_thresholds": {
    "strong_match": ">= 60%",
    "partial_match": ">= 30%",
    "low_match": "< 30%"
  }
}
```

---

### 2. GET /api/skills

Ambil daftar 80 skill yang dikenali sistem. Gunakan untuk dropdown/checklist/autocomplete.

**Response:**
```json
{
  "skills": [
    "accessibility", "adobexd", "agile", "android", "angular",
    "ansible", "api", "appium", "aspnet", "automation", "aws",
    "azure", "bash", "c", "cicd", "cloud", "cpp", "css",
    "database", "design system", "django", "docker", ...
  ],
  "total": 80
}
```

**Catatan:** Cache response ini di frontend. Daftar skill jarang berubah.

---

### 3. POST /api/match (Endpoint Utama)

Ranking lowongan berdasarkan skill user.

**Request:**
```json
{
  "skills": ["react", "javascript", "typescript", "css", "html"],
  "top_n": 10,
  "sort_by": "final_score",
  "min_matched": 1
}
```

| Parameter | Type | Default | Keterangan |
|-----------|------|---------|------------|
| skills | list[str] | (wajib) | Skill user (raw, akan di-normalize otomatis) |
| top_n | int | 10 | Jumlah hasil (1-50) |
| sort_by | str | "final_score" | "final_score", "model_score", atau "coverage_score" |
| min_matched | int | 1 | Minimal skill yang harus match |

**Response:**
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
      "model_score": 0.9341,
      "coverage_score": 0.6667,
      "final_score": 0.7736,
      "match_category": "Strong Match",
      "matched_skills": ["css", "html", "javascript", "react"],
      "missing_skills": ["tailwind", "vue"],
      "n_matched": 4,
      "n_required": 6,
      "location": "jakarta",
      "role": "frontend",
      "roadmap_url": "https://roadmap.sh/frontend"
    }
  ]
}
```

**Match Categories:**
- `Strong Match`: coverage >= 60% (user memenuhi mayoritas skill)
- `Partial Match`: coverage 30-59% (gap moderate)
- `Low Match`: coverage < 30% (gap besar, tapi tetap ditampilkan)

---

### 4. POST /api/predict (Opsional)

Prediksi kecocokan user dengan 1 job spesifik. Gunakan saat user klik detail lowongan.

**Request:**
```json
{
  "user_skills": ["react", "javascript", "css"],
  "job_skills": ["react", "javascript", "css", "typescript", "nextjs", "git"]
}
```

**Response:**
```json
{
  "model_score": 0.9894,
  "coverage_score": 0.5000,
  "final_score": 0.6958,
  "match_category": "Partial Match",
  "matched_skills": ["css", "javascript", "react"],
  "missing_skills": ["git", "nextjs", "typescript"],
  "n_matched": 3,
  "n_required": 6
}
```

**Catatan:** Endpoint ini opsional. Data yang sama sudah tersedia di response `/api/match`. Gunakan `/api/predict` hanya jika perlu refresh atau bandingkan dengan job di luar ranking.

---

### 5. POST /api/normalize (Opsional)

Normalisasi skill input user. Berguna untuk validasi real-time atau setelah OCR.

**Request:**
```json
{
  "skills": ["React.js", "Node.js", "C++", "PostgreSQL", "CI/CD", "UnknownXYZ"]
}
```

**Response:**
```json
{
  "normalized": ["react", "node", "cpp", "sql", "cicd"],
  "unknown": ["UnknownXYZ"],
  "total_recognized": 5,
  "total_unknown": 1
}
```

**Mapping yang dilakukan:**
| Input User | Normalized |
|------------|-----------|
| React.js | react |
| Node.js | node |
| C++ | cpp |
| PostgreSQL | sql |
| CI/CD | cicd |
| Vue.js | vue |
| Next.js | nextjs |
| Spring Boot | springboot |
| Power BI | powerbi |
| JS | javascript |
| TS | typescript |

---

### 6. POST /api/extract-cv

Extract tech skill dari CV (PDF upload).

**Request:** `multipart/form-data`
```
file: [PDF file]
```

**Contoh (JavaScript fetch):**
```javascript
const formData = new FormData();
formData.append('file', pdfFile);

const response = await fetch('/api/extract-cv', {
  method: 'POST',
  body: formData
});
```

**Response:**
```json
{
  "extracted_skills": ["react", "javascript", "node", "sql", "docker", "git"],
  "raw_text_preview": "CURRICULUM VITAE\nNama: Ahmad...",
  "pages_processed": 2,
  "extraction_method": "text_extraction",
  "total_skills_found": 6,
  "direct_matched": ["docker", "git", "javascript", "node", "react", "sql"],
  "fuzzy_matched": [],
  "success": true,
  "error": null
}
```

**Validasi:**
- Format: hanya PDF (`.pdf`)
- Max size: 5MB
- Jika bukan PDF: return 400
- Jika file kosong/corrupt: return 422

**`extraction_method` values:**
- `text_extraction`: PDF berbasis teks (cepat, akurat)
- `ocr_tesseract`: PDF scan/gambar (lebih lambat)

---

## Error Handling

Semua endpoint return HTTP status code yang konsisten:

| Status | Arti | Kapan |
|--------|------|-------|
| 200 | Success | Request berhasil |
| 400 | Bad Request | Input tidak valid (skill kosong, file bukan PDF, dll) |
| 422 | Unprocessable | File tidak bisa diproses (PDF corrupt) |
| 503 | Service Unavailable | Model belum loaded (server baru start) |

**Contoh error response:**
```json
{
  "detail": "No valid skills recognized. Unknown: ['FakeSkill1', 'FakeSkill2']"
}
```

**Rekomendasi frontend:**
- 200: tampilkan data
- 400: tampilkan pesan error ke user
- 503: tampilkan "Loading..." atau retry setelah beberapa detik
- Network error: tampilkan "Tidak bisa terhubung ke server"

---

## Catatan Penting untuk Frontend

1. **`GET /api/skills`** — Panggil sekali saat app load, cache di state/localStorage. Daftar skill jarang berubah.

2. **`POST /api/match`** — Response sudah include semua data yang dibutuhkan untuk list view DAN detail view (`matched_skills`, `missing_skills`, `roadmap_url`). Tidak perlu panggil endpoint lain untuk menampilkan hasil.

3. **`POST /api/predict`** — Opsional. Hanya kalau frontend mau refresh detail atau bandingkan dengan job yang tidak ada di ranking.

4. **`POST /api/extract-cv`** — Kirim sebagai `multipart/form-data` (file upload), BUKAN JSON. Setelah dapat `extracted_skills`, tampilkan ke user untuk review sebelum submit ke `/api/match`.

5. **`POST /api/normalize`** — Opsional. Untuk validasi real-time saat user ketik di input field (autocomplete/suggestion).

6. **Skill input** — User bisa input skill dalam format apapun (React.js, react, REACT). API akan normalize otomatis. Frontend tidak perlu normalize sendiri.

7. **Display message** — API hanya return data (score, category, matched, missing). Frontend yang generate pesan ke user berdasarkan `match_category`:
   - Strong Match → "Lowongan ini cocok untukmu!"
   - Partial Match → "Kamu butuh belajar X skill lagi"
   - Low Match → "Gap besar, tapi roadmap tersedia"

8. **Roadmap URL** — Setiap result punya field `roadmap_url` (link ke roadmap.sh). Frontend tinggal render sebagai link/button.
