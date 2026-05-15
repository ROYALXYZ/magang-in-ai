"""
=============================================================================
FastAPI Application - Magang-in Skill Matching API
=============================================================================
Magang-in Project - AI Team

Side Quest 1: REST API mandiri menggunakan FastAPI untuk melayani model ML.

Endpoints:
  GET  /api/health     - Health check
  GET  /api/skills     - Daftar skill vocabulary
  POST /api/match      - Ranking lowongan berdasarkan skill user
  POST /api/predict    - Prediksi single pair (user vs 1 job)
  POST /api/normalize  - Normalisasi skill input

Cara jalankan:
  pip install fastapi uvicorn
  uvicorn app:app --host 0.0.0.0 --port 8000

Swagger UI (auto-docs):
  http://localhost:8000/docs

Untuk Docker deployment:
  docker build -t magangin-ai .
  docker run -d -p 8000:8000 magangin-ai
=============================================================================
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
import time

# Import SkillMatcher dari inference script
from inference_siamese import SkillMatcher
# Import CVParser dari OCR pipeline
from ocr_pipeline import CVParser


# =============================================================================
# LIFESPAN: Load model saat startup (sekali saja, bukan per-request)
# =============================================================================

matcher: Optional[SkillMatcher] = None
cv_parser: Optional[CVParser] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model saat server start, cleanup saat server stop."""
    global matcher, cv_parser
    print("Starting Magang-in AI API...")
    start_time = time.time()
    matcher = SkillMatcher()
    cv_parser = CVParser(vocab_path='data/skill_vocabulary.csv')
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    yield
    # Cleanup (jika perlu)
    print("Shutting down Magang-in AI API...")


# =============================================================================
# FASTAPI APP INSTANCE
# =============================================================================

app = FastAPI(
    title="Magang-in Skill Matching API",
    description=(
        "REST API untuk mencocokkan skill user dengan lowongan magang. "
        "Menggunakan Siamese Neural Network dengan hybrid scoring "
        "(model similarity + skill coverage)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (agar frontend bisa akses dari domain berbeda)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: ganti dengan domain frontend spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# PYDANTIC MODELS (Request/Response Schema)
# =============================================================================

# --- Request Models ---

class MatchRequest(BaseModel):
    """Request body untuk endpoint /api/match"""
    skills: list[str] = Field(
        ...,
        min_length=1,
        description="List skill user (raw input, akan di-normalize otomatis)",
        examples=[["react", "javascript", "typescript", "css", "html"]]
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Jumlah rekomendasi teratas (1-50)"
    )
    sort_by: str = Field(
        default="final_score",
        description="Sorting: 'final_score', 'model_score', atau 'coverage_score'"
    )
    min_matched: int = Field(
        default=1,
        ge=0,
        description="Minimal jumlah skill yang harus match"
    )


class PredictRequest(BaseModel):
    """Request body untuk endpoint /api/predict"""
    user_skills: list[str] = Field(
        ...,
        min_length=1,
        description="List skill user",
        examples=[["react", "javascript", "css"]]
    )
    job_skills: list[str] = Field(
        ...,
        min_length=1,
        description="List skill yang dibutuhkan job",
        examples=[["react", "javascript", "css", "typescript", "nextjs", "git"]]
    )


class NormalizeRequest(BaseModel):
    """Request body untuk endpoint /api/normalize"""
    skills: list[str] = Field(
        ...,
        min_length=1,
        description="List skill mentah yang ingin di-normalize",
        examples=[["React.js", "Node.js", "C++", "PostgreSQL", "CI/CD"]]
    )


# --- Response Models ---

class JobResult(BaseModel):
    """Schema untuk satu hasil rekomendasi job"""
    rank: int
    title: str
    company: str
    model_score: float
    coverage_score: float
    final_score: float
    match_category: str
    matched_skills: list[str]
    missing_skills: list[str]
    n_matched: int
    n_required: int
    location: str
    role: str
    roadmap_url: str


class MatchResponse(BaseModel):
    """Response body untuk endpoint /api/match"""
    user_skills_normalized: list[str]
    unknown_skills: list[str]
    total_results: int
    results: list[JobResult]


class PredictResponse(BaseModel):
    """Response body untuk endpoint /api/predict"""
    model_score: float
    coverage_score: float
    final_score: float
    match_category: str
    matched_skills: list[str]
    missing_skills: list[str]
    n_matched: int
    n_required: int


class NormalizeResponse(BaseModel):
    """Response body untuk endpoint /api/normalize"""
    normalized: list[str]
    unknown: list[str]
    total_recognized: int
    total_unknown: int


class HealthResponse(BaseModel):
    """Response body untuk endpoint /api/health"""
    status: str
    model_loaded: bool
    vocab_size: int
    jobs_count: int
    scoring_formula: str
    match_thresholds: dict


class SkillsResponse(BaseModel):
    """Response body untuk endpoint /api/skills"""
    skills: list[str]
    total: int


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Cek apakah API hidup dan model sudah loaded.
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    return HealthResponse(
        status="ok",
        model_loaded=True,
        vocab_size=matcher.vocab_size,
        jobs_count=len(matcher.jobs_df),
        scoring_formula=f"{matcher.MODEL_WEIGHT}*model + {matcher.COVERAGE_WEIGHT}*coverage",
        match_thresholds={
            "strong_match": f">= {matcher.STRONG_MATCH_THRESHOLD*100:.0f}%",
            "partial_match": f">= {matcher.PARTIAL_MATCH_THRESHOLD*100:.0f}%",
            "low_match": f"< {matcher.PARTIAL_MATCH_THRESHOLD*100:.0f}%"
        }
    )


@app.get("/api/skills", response_model=SkillsResponse, tags=["Skills"])
async def get_skills():
    """
    Daftar semua skill yang dikenali sistem.
    Gunakan untuk dropdown/autocomplete di frontend.
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    skills = matcher.get_skill_vocabulary()
    return SkillsResponse(skills=skills, total=len(skills))


@app.post("/api/match", response_model=MatchResponse, tags=["Matching"])
async def match_skills(request: MatchRequest):
    """
    Ranking lowongan berdasarkan kecocokan skill user.

    Scoring: final_score = 0.4 * model_score + 0.6 * coverage_score

    Match Categories (berdasarkan coverage):
    - Strong Match: coverage >= 60%
    - Partial Match: coverage 30-59%
    - Low Match: coverage < 30%

    Filter: minimal 1 skill match (default). Job tanpa skill match tidak ditampilkan.
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    # Validate sort_by
    valid_sort = ["final_score", "model_score", "coverage_score"]
    if request.sort_by not in valid_sort:
        raise HTTPException(
            status_code=400,
            detail=f"sort_by must be one of: {valid_sort}"
        )

    # Normalize skills
    normalized, unknown = matcher.normalize_user_skills(request.skills)

    if not normalized:
        raise HTTPException(
            status_code=400,
            detail=f"No valid skills recognized. Unknown: {unknown}"
        )

    # Get ranking
    results = matcher.rank_jobs(
        user_skills=request.skills,
        top_n=request.top_n,
        min_matched=request.min_matched,
        sort_by=request.sort_by
    )

    # Convert to response format
    job_results = [
        JobResult(
            rank=r['rank'],
            title=r['title'],
            company=r['company'],
            model_score=round(r['model_score'], 4),
            coverage_score=round(r['coverage_score'], 4),
            final_score=round(r['final_score'], 4),
            match_category=r['match_category'],
            matched_skills=r['matched_skills'],
            missing_skills=r['missing_skills'],
            n_matched=r['n_matched'],
            n_required=r['n_required'],
            location=r['location'],
            role=r['role'],
            roadmap_url=r['roadmap_url'],
        )
        for r in results
    ]

    return MatchResponse(
        user_skills_normalized=normalized,
        unknown_skills=unknown,
        total_results=len(job_results),
        results=job_results
    )


@app.post("/api/predict", response_model=PredictResponse, tags=["Matching"])
async def predict_match(request: PredictRequest):
    """
    Prediksi kecocokan antara skill user dan skill satu job spesifik.
    Gunakan untuk detail view ketika user klik satu lowongan.
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    result = matcher.predict_match(request.user_skills, request.job_skills)

    return PredictResponse(
        model_score=round(result['model_score'], 4),
        coverage_score=round(result['coverage_score'], 4),
        final_score=round(result['final_score'], 4),
        match_category=result['match_category'],
        matched_skills=result['matched_skills'],
        missing_skills=result['missing_skills'],
        n_matched=result['n_matched'],
        n_required=result['n_required'],
    )


@app.post("/api/normalize", response_model=NormalizeResponse, tags=["Skills"])
async def normalize_skills(request: NormalizeRequest):
    """
    Normalisasi skill input user.
    Berguna untuk validasi sebelum submit, atau setelah OCR extract.

    Contoh normalisasi:
    - "React.js" → "react"
    - "Node.js" → "node"
    - "C++" → "cpp"
    - "PostgreSQL" → "sql"
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    normalized, unknown = matcher.normalize_user_skills(request.skills)

    return NormalizeResponse(
        normalized=normalized,
        unknown=unknown,
        total_recognized=len(normalized),
        total_unknown=len(unknown),
    )


# --- Response Model untuk Extract CV ---

class ExtractCVResponse(BaseModel):
    """Response body untuk endpoint /api/extract-cv"""
    extracted_skills: list[str]
    raw_text_preview: str
    pages_processed: int
    extraction_method: str
    total_skills_found: int
    direct_matched: list[str]
    fuzzy_matched: list[str]
    success: bool
    error: Optional[str]


@app.post("/api/extract-cv", response_model=ExtractCVResponse, tags=["CV Parsing"])
async def extract_cv_skills(file: UploadFile = File(...)):
    """
    Extract tech skills dari CV (PDF).

    Pipeline:
    1. Upload file PDF
    2. Extract teks (PyMuPDF untuk PDF teks, Tesseract OCR untuk PDF scan)
    3. Fuzzy matching dengan skill vocabulary (threshold 80%)
    4. Return daftar skill terdeteksi

    Supported format: PDF only.
    Max file size: 5MB.

    Output bisa langsung dipakai sebagai input untuk /api/match.
    """
    if cv_parser is None:
        raise HTTPException(status_code=503, detail="CV Parser not ready")

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )

    # Validate file size (max 5MB)
    contents = await file.read()
    max_size = 5 * 1024 * 1024  # 5MB
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is 5MB. Your file: {len(contents)/1024/1024:.1f}MB"
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="File is empty."
        )

    # Extract skills from PDF bytes
    result = cv_parser.extract_skills_from_pdf(pdf_bytes=contents)

    if not result['success']:
        raise HTTPException(
            status_code=422,
            detail=result.get('error', 'Failed to extract text from PDF.')
        )

    return ExtractCVResponse(
        extracted_skills=result['extracted_skills'],
        raw_text_preview=result['raw_text_preview'],
        pages_processed=result['pages_processed'],
        extraction_method=result['extraction_method'],
        total_skills_found=result['total_skills_found'],
        direct_matched=result['direct_matched'],
        fuzzy_matched=result['fuzzy_matched'],
        success=result['success'],
        error=result['error'],
    )


# =============================================================================
# MAIN (untuk development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
