# Frontend Integration Guide — Magang-in AI API

Panduan integrasi API Magang-in AI untuk tim fullstack (React + TypeScript + Express).

## Arsitektur

```
Browser (React + TypeScript)
    │
    │ fetch('/api/match', ...)
    ▼
Express Backend (proxy)
    │
    │ fetch('https://api-magangin.domain.com/api/match', ...)
    ▼
FastAPI (Magang-in AI)
    │
    │ model.predict()
    ▼
Response JSON
```

Semua request dari React melewati Express backend sebagai proxy. Express meneruskan request ke API Magang-in AI.

---

## Setup

### Base URL

```typescript
// Di Express backend (.env)
MAGANGIN_AI_URL=https://api-magangin.domain.com

// Atau untuk development lokal (jika VPS di jaringan yang sama)
MAGANGIN_AI_URL=http://192.168.30.3:8001
```

---

## Express Backend — Proxy Routes

### File: `routes/ai.ts`

```typescript
import { Router, Request, Response } from 'express';

const router = Router();

// Base URL API Magang-in AI
const AI_API_URL = process.env.MAGANGIN_AI_URL || 'http://192.168.30.3:8001';

/**
 * GET /api/ai/health
 * Cek apakah AI API hidup
 */
router.get('/health', async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${AI_API_URL}/api/health`);
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(503).json({ error: 'AI service unavailable' });
  }
});

/**
 * GET /api/ai/skills
 * Ambil daftar skill vocabulary (80 skill)
 */
router.get('/skills', async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${AI_API_URL}/api/skills`);
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(503).json({ error: 'AI service unavailable' });
  }
});

/**
 * POST /api/ai/match
 * Ranking lowongan berdasarkan skill user
 */
router.post('/match', async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${AI_API_URL}/api/match`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json(data);
    }

    res.json(data);
  } catch (error) {
    res.status(503).json({ error: 'AI service unavailable' });
  }
});

/**
 * POST /api/ai/predict
 * Prediksi kecocokan user vs 1 job spesifik
 */
router.post('/predict', async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${AI_API_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json(data);
    }

    res.json(data);
  } catch (error) {
    res.status(503).json({ error: 'AI service unavailable' });
  }
});

/**
 * POST /api/ai/normalize
 * Normalisasi skill input
 */
router.post('/normalize', async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${AI_API_URL}/api/normalize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json(data);
    }

    res.json(data);
  } catch (error) {
    res.status(503).json({ error: 'AI service unavailable' });
  }
});

/**
 * POST /api/ai/extract-cv
 * Upload CV (PDF) dan extract skill
 * Note: Ini forward multipart/form-data
 */
router.post('/extract-cv', async (req: Request, res: Response) => {
  try {
    // Forward file as-is ke AI API
    // Gunakan library seperti 'form-data' atau 'multer' untuk handle file
    // Contoh dengan multer (file sudah di req.file):

    const formData = new FormData();
    formData.append('file', new Blob([req.file!.buffer]), req.file!.originalname);

    const response = await fetch(`${AI_API_URL}/api/extract-cv`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json(data);
    }

    res.json(data);
  } catch (error) {
    res.status(503).json({ error: 'AI service unavailable' });
  }
});

export default router;
```

### File: `app.ts` (register route)

```typescript
import express from 'express';
import aiRoutes from './routes/ai';
import multer from 'multer';

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.use(express.json());

// AI routes (proxy ke Magang-in AI API)
app.use('/api/ai', aiRoutes);

// Khusus extract-cv (perlu multer untuk handle file upload)
app.post('/api/ai/extract-cv', upload.single('file'), aiRoutes);

app.listen(5000, () => {
  console.log('Express server running on port 5000');
});
```

---

## React Frontend — TypeScript

### Types

```typescript
// types/ai.ts

export interface MatchRequest {
  skills: string[];
  top_n?: number;
  sort_by?: 'final_score' | 'model_score' | 'coverage_score';
  min_matched?: number;
}

export interface JobResult {
  rank: number;
  title: string;
  company: string;
  model_score: number;
  coverage_score: number;
  final_score: number;
  match_category: 'Strong Match' | 'Partial Match' | 'Low Match';
  matched_skills: string[];
  missing_skills: string[];
  n_matched: number;
  n_required: number;
  location: string;
  role: string;
  roadmap_url: string;
}

export interface MatchResponse {
  user_skills_normalized: string[];
  unknown_skills: string[];
  total_results: number;
  results: JobResult[];
}

export interface PredictRequest {
  user_skills: string[];
  job_skills: string[];
}

export interface PredictResponse {
  model_score: number;
  coverage_score: number;
  final_score: number;
  match_category: 'Strong Match' | 'Partial Match' | 'Low Match';
  matched_skills: string[];
  missing_skills: string[];
  n_matched: number;
  n_required: number;
}

export interface NormalizeRequest {
  skills: string[];
}

export interface NormalizeResponse {
  normalized: string[];
  unknown: string[];
  total_recognized: number;
  total_unknown: number;
}

export interface ExtractCVResponse {
  extracted_skills: string[];
  raw_text_preview: string;
  pages_processed: number;
  extraction_method: 'text_extraction' | 'ocr_tesseract';
  total_skills_found: number;
  direct_matched: string[];
  fuzzy_matched: string[];
  success: boolean;
  error: string | null;
}

export interface SkillsResponse {
  skills: string[];
  total: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  vocab_size: number;
  jobs_count: number;
  scoring_formula: string;
  match_thresholds: {
    strong_match: string;
    partial_match: string;
    low_match: string;
  };
}
```

---

### API Service

```typescript
// services/aiService.ts

import type {
  MatchRequest,
  MatchResponse,
  PredictRequest,
  PredictResponse,
  NormalizeRequest,
  NormalizeResponse,
  ExtractCVResponse,
  SkillsResponse,
  HealthResponse,
} from '../types/ai';

const API_BASE = '/api/ai'; // Proxy lewat Express

/**
 * Cek apakah AI service hidup
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error('AI service unavailable');
  return response.json();
}

/**
 * Ambil daftar skill vocabulary (untuk dropdown/checklist)
 */
export async function getSkills(): Promise<SkillsResponse> {
  const response = await fetch(`${API_BASE}/skills`);
  if (!response.ok) throw new Error('Failed to load skills');
  return response.json();
}

/**
 * Ranking lowongan berdasarkan skill user
 */
export async function matchSkills(request: MatchRequest): Promise<MatchResponse> {
  const response = await fetch(`${API_BASE}/match`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Match failed');
  }

  return response.json();
}

/**
 * Prediksi kecocokan user vs 1 job
 */
export async function predictMatch(request: PredictRequest): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Predict failed');
  }

  return response.json();
}

/**
 * Normalisasi skill input
 */
export async function normalizeSkills(request: NormalizeRequest): Promise<NormalizeResponse> {
  const response = await fetch(`${API_BASE}/normalize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Normalize failed');
  }

  return response.json();
}

/**
 * Extract skill dari CV (PDF upload)
 */
export async function extractCV(file: File): Promise<ExtractCVResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/extract-cv`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'CV extraction failed');
  }

  return response.json();
}
```

---

### React Components

#### Skill Selector (Self-Declaration)

```tsx
// components/SkillSelector.tsx

import { useState, useEffect } from 'react';
import { getSkills } from '../services/aiService';

interface SkillSelectorProps {
  onSubmit: (skills: string[]) => void;
}

export function SkillSelector({ onSubmit }: SkillSelectorProps) {
  const [availableSkills, setAvailableSkills] = useState<string[]>([]);
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSkills()
      .then(data => {
        setAvailableSkills(data.skills);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load skills:', err);
        setLoading(false);
      });
  }, []);

  const toggleSkill = (skill: string) => {
    setSelectedSkills(prev =>
      prev.includes(skill)
        ? prev.filter(s => s !== skill)
        : [...prev, skill]
    );
  };

  if (loading) return <div>Loading skills...</div>;

  return (
    <div>
      <h3>Pilih Skill Kamu</h3>
      <p>{selectedSkills.length} skill dipilih</p>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        {availableSkills.map(skill => (
          <button
            key={skill}
            onClick={() => toggleSkill(skill)}
            style={{
              padding: '6px 12px',
              borderRadius: '16px',
              border: '1px solid #ddd',
              background: selectedSkills.includes(skill) ? '#3498db' : '#fff',
              color: selectedSkills.includes(skill) ? '#fff' : '#333',
              cursor: 'pointer',
            }}
          >
            {skill}
          </button>
        ))}
      </div>

      <button
        onClick={() => onSubmit(selectedSkills)}
        disabled={selectedSkills.length === 0}
        style={{ marginTop: '16px', padding: '12px 24px' }}
      >
        Cari Lowongan Cocok ({selectedSkills.length} skill)
      </button>
    </div>
  );
}
```

#### CV Upload

```tsx
// components/CVUpload.tsx

import { useState } from 'react';
import { extractCV } from '../services/aiService';
import type { ExtractCVResponse } from '../types/ai';

interface CVUploadProps {
  onSkillsExtracted: (skills: string[]) => void;
}

export function CVUpload({ onSkillsExtracted }: CVUploadProps) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExtractCVResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.pdf')) {
      setError('Hanya file PDF yang didukung');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await extractCV(file);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload gagal');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h3>Upload CV (PDF)</h3>

      <input
        type="file"
        accept=".pdf"
        onChange={handleUpload}
        disabled={loading}
      />

      {loading && <p>Mengekstrak skill dari CV...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}

      {result && (
        <div>
          <p>
            {result.total_skills_found} skill terdeteksi
            (method: {result.extraction_method})
          </p>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {result.extracted_skills.map(skill => (
              <span
                key={skill}
                style={{
                  padding: '4px 10px',
                  background: '#d4efdf',
                  borderRadius: '12px',
                  fontSize: '13px',
                }}
              >
                {skill}
              </span>
            ))}
          </div>

          <button
            onClick={() => onSkillsExtracted(result.extracted_skills)}
            style={{ marginTop: '12px', padding: '10px 20px' }}
          >
            Gunakan Skill Ini untuk Matching →
          </button>
        </div>
      )}
    </div>
  );
}
```

#### Match Results

```tsx
// components/MatchResults.tsx

import { useState } from 'react';
import { matchSkills } from '../services/aiService';
import type { MatchResponse, JobResult } from '../types/ai';

interface MatchResultsProps {
  skills: string[];
}

export function MatchResults({ skills }: MatchResultsProps) {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<MatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const doMatch = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await matchSkills({ skills, top_n: 10 });
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Matching gagal');
    } finally {
      setLoading(false);
    }
  };

  // Auto-fetch saat component mount atau skills berubah
  useState(() => { doMatch(); });

  if (loading) return <div>Mencari lowongan yang cocok...</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;
  if (!data) return null;

  return (
    <div>
      <p>
        Skill kamu: {data.user_skills_normalized.join(', ')}
        {data.unknown_skills.length > 0 && (
          <span style={{ color: 'orange' }}>
            {' '}(tidak dikenali: {data.unknown_skills.join(', ')})
          </span>
        )}
      </p>

      <h3>{data.total_results} Lowongan Ditemukan</h3>

      {data.results.map(job => (
        <JobCard key={job.rank} job={job} />
      ))}
    </div>
  );
}

function JobCard({ job }: { job: JobResult }) {
  const coveragePct = Math.round(job.coverage_score * 100);

  // Generate display message berdasarkan category
  const getMessage = () => {
    switch (job.match_category) {
      case 'Strong Match':
        return 'Lowongan ini cocok untukmu!';
      case 'Partial Match':
        return `Belajar ${job.missing_skills.length} skill lagi: ${job.missing_skills.slice(0, 3).join(', ')}`;
      case 'Low Match':
        return `Gap besar (${job.missing_skills.length} skill), tapi roadmap tersedia`;
    }
  };

  // Warna berdasarkan category
  const getColor = () => {
    switch (job.match_category) {
      case 'Strong Match': return '#27ae60';
      case 'Partial Match': return '#f39c12';
      case 'Low Match': return '#e74c3c';
    }
  };

  return (
    <div
      style={{
        border: `1px solid #eee`,
        borderLeft: `4px solid ${getColor()}`,
        borderRadius: '8px',
        padding: '16px',
        marginBottom: '12px',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <strong>#{job.rank} {job.title}</strong>
        <span style={{ color: getColor(), fontWeight: 600 }}>
          {coveragePct}% match
        </span>
      </div>

      <p style={{ color: '#666', fontSize: '14px' }}>
        {job.company} · {job.location} · {job.role}
      </p>

      <p style={{ fontSize: '14px', color: getColor() }}>
        {getMessage()}
      </p>

      {/* Matched skills */}
      <div style={{ marginTop: '8px' }}>
        <strong style={{ fontSize: '12px' }}>Matched:</strong>
        {job.matched_skills.map(s => (
          <span key={s} style={{ padding: '2px 8px', background: '#d4efdf', borderRadius: '4px', fontSize: '12px', margin: '2px' }}>
            {s}
          </span>
        ))}
      </div>

      {/* Missing skills */}
      <div style={{ marginTop: '4px' }}>
        <strong style={{ fontSize: '12px' }}>Missing:</strong>
        {job.missing_skills.map(s => (
          <span key={s} style={{ padding: '2px 8px', background: '#fadbd8', borderRadius: '4px', fontSize: '12px', margin: '2px' }}>
            {s}
          </span>
        ))}
      </div>

      {/* Roadmap link */}
      {job.roadmap_url && (
        <a
          href={job.roadmap_url}
          target="_blank"
          rel="noopener noreferrer"
          style={{ fontSize: '13px', color: '#3498db', marginTop: '8px', display: 'block' }}
        >
          📚 Lihat Roadmap →
        </a>
      )}
    </div>
  );
}
```

#### Main Page (Gabungan)

```tsx
// pages/MatchingPage.tsx

import { useState } from 'react';
import { SkillSelector } from '../components/SkillSelector';
import { CVUpload } from '../components/CVUpload';
import { MatchResults } from '../components/MatchResults';

export function MatchingPage() {
  const [userSkills, setUserSkills] = useState<string[] | null>(null);
  const [inputMethod, setInputMethod] = useState<'self' | 'cv' | null>(null);

  // Kalau user sudah submit skill, tampilkan hasil
  if (userSkills && userSkills.length > 0) {
    return (
      <div>
        <button onClick={() => setUserSkills(null)}>← Ubah Skill</button>
        <MatchResults skills={userSkills} />
      </div>
    );
  }

  return (
    <div>
      <h1>Magang-in — Cari Magang yang Cocok</h1>

      {!inputMethod && (
        <div>
          <h2>Bagaimana kamu ingin memasukkan skill?</h2>
          <button onClick={() => setInputMethod('self')}>
            Pilih Manual (Self-Declaration)
          </button>
          <button onClick={() => setInputMethod('cv')}>
            Upload CV (PDF)
          </button>
        </div>
      )}

      {inputMethod === 'self' && (
        <SkillSelector onSubmit={setUserSkills} />
      )}

      {inputMethod === 'cv' && (
        <CVUpload onSkillsExtracted={setUserSkills} />
      )}
    </div>
  );
}
```

---

## Error Handling Pattern

```typescript
// utils/errorHandler.ts

export async function handleApiError(response: Response): Promise<never> {
  const data = await response.json().catch(() => ({}));

  switch (response.status) {
    case 400:
      throw new Error(data.detail || 'Input tidak valid');
    case 422:
      throw new Error(data.detail || 'File tidak bisa diproses');
    case 503:
      throw new Error('AI service sedang tidak tersedia. Coba lagi nanti.');
    default:
      throw new Error(data.detail || 'Terjadi kesalahan');
  }
}
```

---

## Loading State Pattern

```tsx
// hooks/useAiMatch.ts

import { useState } from 'react';
import { matchSkills } from '../services/aiService';
import type { MatchResponse } from '../types/ai';

export function useAiMatch() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<MatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const doMatch = async (skills: string[]) => {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const result = await matchSkills({ skills, top_n: 10 });
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return { loading, data, error, doMatch };
}

// Penggunaan di component:
// const { loading, data, error, doMatch } = useAiMatch();
// <button onClick={() => doMatch(['react', 'javascript'])}>Match</button>
```

---

## Caching Pattern

```typescript
// services/skillCache.ts

let cachedSkills: string[] | null = null;

export async function getCachedSkills(): Promise<string[]> {
  if (cachedSkills) return cachedSkills;

  const response = await fetch('/api/ai/skills');
  const data = await response.json();
  cachedSkills = data.skills;
  return cachedSkills;
}
```

Skill vocabulary jarang berubah (80 skill fixed). Cache di memory atau localStorage supaya tidak fetch berulang kali.

---

## Tips & Best Practices

1. **Cache `/api/skills`** — Panggil sekali saat app load, simpan di state/context/localStorage.

2. **Debounce normalize** — Kalau pakai `/api/normalize` untuk autocomplete, debounce 300ms supaya tidak spam API.

3. **Loading state** — Selalu tampilkan loading indicator saat menunggu response. `/api/match` bisa butuh 1-2 detik.

4. **Handle unknown skills** — Response `/api/match` punya field `unknown_skills`. Tampilkan warning ke user: "Skill X tidak dikenali oleh sistem."

5. **Roadmap URL** — Setiap job result punya `roadmap_url`. Render sebagai link external (`target="_blank"`).

6. **Display message di frontend** — API hanya return data + `match_category`. Frontend yang generate pesan motivasi berdasarkan category:
   - Strong Match → hijau, pesan positif
   - Partial Match → kuning, pesan "belajar X skill lagi"
   - Low Match → merah, pesan "gap besar tapi roadmap tersedia"

7. **File upload (CV)** — Gunakan `FormData`, bukan JSON. Max 5MB. Hanya PDF.

8. **Error 503** — Artinya model belum loaded (server baru start). Retry setelah 10 detik atau tampilkan "Sedang memuat model AI..."
