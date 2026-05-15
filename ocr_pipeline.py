"""
=============================================================================
OCR PIPELINE - CV Parsing for Skill Extraction
=============================================================================
Magang-in Project - AI Team

Pipeline untuk mengekstrak tech skill dari CV (PDF).
Strategi dual:
  1. Primary: PyMuPDF (fitz) — extract teks langsung dari PDF berbasis teks
  2. Fallback: PyTesseract OCR — untuk PDF berbasis gambar (scan)

Setelah teks diekstrak, dilakukan fuzzy matching terhadap skill vocabulary
untuk mengidentifikasi tech skill yang dimiliki user.

Cara pakai:
    from ocr_pipeline import CVParser
    parser = CVParser(vocab_path='data/skill_vocabulary.csv')
    result = parser.extract_skills_from_pdf('cv_user.pdf')
=============================================================================
"""

import re
import os
import pandas as pd
import numpy as np
from typing import Optional

# PDF text extraction
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. Install with: pip install PyMuPDF")

# OCR fallback
try:
    from pdf2image import convert_from_path, convert_from_bytes
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: Tesseract/pdf2image not installed. OCR fallback disabled.")

# Fuzzy matching
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not installed. Install with: pip install rapidfuzz")


class CVParser:
    """
    Parser CV (PDF) untuk mengekstrak tech skill.

    Pipeline:
        PDF → Text Extraction (PyMuPDF/Tesseract) → Cleaning → Fuzzy Matching → Skill List

    Args:
        vocab_path: path ke skill_vocabulary.csv
        fuzzy_threshold: minimum similarity score untuk fuzzy match (0-100)
        min_text_length: minimum panjang teks dari PyMuPDF sebelum fallback ke OCR
    """

    # Minimum panjang teks agar dianggap valid (bukan PDF scan)
    MIN_TEXT_LENGTH = 50

    # Fuzzy matching threshold (0-100)
    DEFAULT_FUZZY_THRESHOLD = 80

    # Skill normalization mapping (sama dengan inference_siamese.py)
    SKILL_NORMALIZE_MAP = {
        'ci/cd': 'cicd',
        'ci cd': 'cicd',
        'c++': 'cpp',
        'c#': 'csharp',
        'node.js': 'node',
        'nodejs': 'node',
        'react.js': 'react',
        'reactjs': 'react',
        'vue.js': 'vue',
        'vuejs': 'vue',
        'next.js': 'nextjs',
        'express.js': 'express',
        'spring boot': 'springboot',
        'react-native': 'react native',
        'power bi': 'powerbi',
        'adobe xd': 'adobexd',
        'postgresql': 'sql',
        'mysql': 'sql',
        'sql server': 'sql',
        'mongo': 'mongodb',
        'mongodb': 'mongodb',
        'k8s': 'kubernetes',
        'amazon web services': 'aws',
        'google cloud': 'gcp',
        'github': 'git',
        'gitlab': 'git',
        # Singkatan umum
        'js': 'javascript',
        'ts': 'typescript',
        'py': 'python',
        'rb': 'ruby on rails',
        'pg': 'sql',
        'tf': 'tensorflow',
        'pt': 'pytorch',
        'np': 'numpy',
        'pd': 'pandas',
        # Variasi spasi/dash
        'post man': 'postman',
        'vue js': 'vue',
        'vue.js': 'vue',
        'react js': 'react',
        'node js': 'node',
        'next js': 'nextjs',
        'type script': 'typescript',
        'java script': 'javascript',
        'spring-boot': 'springboot',
        'fast-api': 'fastapi',
        'fast api': 'fastapi',
        'tensor flow': 'tensorflow',
        'tensor-flow': 'tensorflow',
        'py-torch': 'pytorch',
        'sci-kit learn': 'sklearn',
        'scikit-learn': 'sklearn',
        'scikit learn': 'sklearn',
        'mongo db': 'mongodb',
        'type-script': 'typescript',
    }

    # Mapping istilah Bahasa Indonesia ke skill vocabulary
    INDONESIAN_SKILL_MAP = {
        # Database & Data
        'basis data': 'database',
        'pangkalan data': 'database',
        'manajemen basis data': 'database',
        'analisis data': 'pandas',
        'visualisasi data': 'tableau',
        'ilmu data': 'python',
        'rekayasa data': 'sql',
        # Networking & Security
        'jaringan': 'networking',
        'jaringan komputer': 'networking',
        'keamanan': 'security',
        'keamanan siber': 'security',
        'keamanan jaringan': 'network security',
        'keamanan informasi': 'security',
        'uji penetrasi': 'penetration',
        # Development
        'pengembangan web': 'html',
        'pengembangan aplikasi': 'javascript',
        'pengembangan mobile': 'flutter',
        'pengembangan frontend': 'react',
        'pengembangan backend': 'node',
        'antarmuka pengguna': 'figma',
        'desain antarmuka': 'figma',
        'pengalaman pengguna': 'user research',
        # Tools & Practices
        'kontrol versi': 'git',
        'pengendalian versi': 'git',
        'komputasi awan': 'cloud',
        'layanan awan': 'cloud',
        'wadah': 'docker',
        'kontainerisasi': 'docker',
        'otomatisasi': 'automation',
        'pengujian': 'testing',
        'pengujian perangkat lunak': 'testing',
        'jaminan kualitas': 'quality assurance',
        'penjaminan mutu': 'quality assurance',
        # AI/ML
        'pembelajaran mesin': 'sklearn',
        'pembelajaran mendalam': 'tensorflow',
        'kecerdasan buatan': 'python',
        'jaringan saraf': 'tensorflow',
        'pengolahan bahasa alami': 'python',
        'visi komputer': 'python',
        # OS
        'sistem operasi linux': 'linux',
    }

    def __init__(self, vocab_path='data/skill_vocabulary.csv',
                 fuzzy_threshold=None):
        """
        Initialize CVParser.

        Args:
            vocab_path: path ke skill_vocabulary.csv
            fuzzy_threshold: threshold fuzzy matching (default 80)
        """
        self.fuzzy_threshold = fuzzy_threshold or self.DEFAULT_FUZZY_THRESHOLD

        # Load vocabulary
        vocab_df = pd.read_csv(vocab_path)
        self.skill_list = vocab_df['skill'].tolist()
        self.skill_to_idx = {skill: idx for idx, skill in enumerate(self.skill_list)}
        self.vocab_size = len(self.skill_list)

        # Gabungkan semua mapping untuk lookup cepat
        self.all_mappings = {}
        self.all_mappings.update(self.SKILL_NORMALIZE_MAP)
        self.all_mappings.update(self.INDONESIAN_SKILL_MAP)

        print(f"CVParser initialized:")
        print(f"  Vocabulary: {self.vocab_size} skills")
        print(f"  Fuzzy threshold: {self.fuzzy_threshold}%")
        print(f"  PyMuPDF available: {PYMUPDF_AVAILABLE}")
        print(f"  Tesseract available: {TESSERACT_AVAILABLE}")
        print(f"  RapidFuzz available: {RAPIDFUZZ_AVAILABLE}")
        print()

    # =========================================================================
    # TEXT EXTRACTION
    # =========================================================================

    def _extract_text_pymupdf(self, pdf_path=None, pdf_bytes=None):
        """
        Extract teks dari PDF menggunakan PyMuPDF.
        Untuk PDF berbasis teks (dibuat dari Word/Canva/dll).

        Args:
            pdf_path: path ke file PDF
            pdf_bytes: bytes content PDF (untuk upload via API)

        Returns:
            tuple: (text, n_pages)
        """
        if not PYMUPDF_AVAILABLE:
            return "", 0

        try:
            if pdf_bytes:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            else:
                doc = fitz.open(pdf_path)

            text = ""
            n_pages = len(doc)

            for page in doc:
                text += page.get_text() + "\n"

            doc.close()
            return text.strip(), n_pages

        except Exception as e:
            print(f"  PyMuPDF error: {e}")
            return "", 0

    def _extract_text_tesseract(self, pdf_path=None, pdf_bytes=None):
        """
        Extract teks dari PDF menggunakan Tesseract OCR.
        Fallback untuk PDF berbasis gambar (scan).

        Args:
            pdf_path: path ke file PDF
            pdf_bytes: bytes content PDF

        Returns:
            tuple: (text, n_pages)
        """
        if not TESSERACT_AVAILABLE:
            return "", 0

        try:
            # Convert PDF pages ke images
            if pdf_bytes:
                images = convert_from_bytes(pdf_bytes)
            else:
                images = convert_from_path(pdf_path)

            n_pages = len(images)
            text = ""

            for i, image in enumerate(images):
                # OCR setiap halaman
                page_text = pytesseract.image_to_string(
                    image, lang='eng+ind'  # English + Indonesian
                )
                text += page_text + "\n"

            return text.strip(), n_pages

        except Exception as e:
            print(f"  Tesseract error: {e}")
            return "", 0

    def extract_text(self, pdf_path=None, pdf_bytes=None):
        """
        Extract teks dari PDF dengan dual strategy:
        1. Coba PyMuPDF dulu (cepat, untuk PDF teks)
        2. Kalau gagal/kosong, fallback ke Tesseract (untuk PDF scan)

        Args:
            pdf_path: path ke file PDF
            pdf_bytes: bytes content PDF

        Returns:
            dict: {text, n_pages, method}
        """
        # Strategy 1: PyMuPDF (text extraction)
        text, n_pages = self._extract_text_pymupdf(pdf_path, pdf_bytes)

        if len(text) >= self.MIN_TEXT_LENGTH:
            return {
                'text': text,
                'n_pages': n_pages,
                'method': 'text_extraction',
                'success': True
            }

        # Strategy 2: Tesseract OCR (fallback)
        print("  PyMuPDF returned insufficient text. Trying Tesseract OCR...")
        text_ocr, n_pages_ocr = self._extract_text_tesseract(pdf_path, pdf_bytes)

        if len(text_ocr) >= self.MIN_TEXT_LENGTH:
            return {
                'text': text_ocr,
                'n_pages': n_pages_ocr,
                'method': 'ocr_tesseract',
                'success': True
            }

        # Both failed
        # Return whatever we got (might be empty or very short)
        final_text = text if len(text) > len(text_ocr) else text_ocr
        final_pages = n_pages if n_pages > 0 else n_pages_ocr

        return {
            'text': final_text,
            'n_pages': final_pages,
            'method': 'failed',
            'success': len(final_text) > 0
        }

    # =========================================================================
    # TEXT CLEANING & PREPROCESSING
    # =========================================================================

    def _clean_text(self, text):
        """
        Bersihkan teks hasil extraction/OCR.

        Steps:
        1. Lowercase
        2. Ganti newline/tab dengan spasi
        3. Hapus URL
        4. Hapus email
        5. Hapus nomor telepon
        6. Hapus special characters (kecuali yang relevan: +, #, .)
        7. Collapse multiple spaces
        """
        # Lowercase
        text = text.lower()

        # Ganti newline/tab dengan spasi
        text = re.sub(r'[\n\r\t]+', ' ', text)

        # Hapus URL
        text = re.sub(r'https?://\S+', ' ', text)
        text = re.sub(r'www\.\S+', ' ', text)

        # Hapus email
        text = re.sub(r'\S+@\S+\.\S+', ' ', text)

        # Hapus nomor telepon (format Indonesia)
        text = re.sub(r'(\+62|62|0)\d{8,12}', ' ', text)

        # Pertahankan karakter yang relevan untuk skill: +, #, .
        # Hapus special char lainnya
        text = re.sub(r'[^\w\s\+\#\.\-\/]', ' ', text)

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _tokenize(self, text):
        """
        Tokenize teks menjadi words dan phrases (unigram + bigram).

        Returns:
            tuple: (words, bigrams)
        """
        words = text.split()

        # Generate bigrams untuk multi-word skills
        bigrams = []
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]} {words[i+1]}")

        return words, bigrams

    # =========================================================================
    # SKILL MATCHING
    # =========================================================================

    def _direct_match(self, tokens):
        """
        Layer 1: Direct/exact match setelah normalisasi.
        Cek apakah token ada di vocabulary atau mapping.
        Juga strip angka di akhir (Python3 → python, HTML5 → html).
        """
        matched = set()

        for token in tokens:
            # Cek di normalize map langsung
            if token in self.all_mappings:
                skill = self.all_mappings[token]
                if skill in self.skill_to_idx:
                    matched.add(skill)
            # Cek langsung di vocabulary
            elif token in self.skill_to_idx:
                matched.add(token)
            else:
                # Strip angka/versi di akhir: Python3 → python, HTML5 → html
                token_stripped = re.sub(r'[\d\.]+$', '', token).strip()
                if token_stripped and token_stripped != token:
                    if token_stripped in self.all_mappings:
                        skill = self.all_mappings[token_stripped]
                        if skill in self.skill_to_idx:
                            matched.add(skill)
                    elif token_stripped in self.skill_to_idx:
                        matched.add(token_stripped)

        return matched

    def _fuzzy_match(self, tokens, already_matched):
        """
        Layer 2: Fuzzy matching untuk menangkap typo dan variasi.
        Hanya untuk token yang belum ter-match di Layer 1.
        """
        if not RAPIDFUZZ_AVAILABLE:
            return set()

        matched = set()
        unmatched_tokens = [t for t in tokens if t not in already_matched
                           and t not in self.all_mappings
                           and t not in self.skill_to_idx]

        for token in unmatched_tokens:
            # Skip token terlalu pendek (< 2 char) atau terlalu panjang (> 25 char)
            if len(token) < 2 or len(token) > 25:
                continue

            # Skip token yang jelas bukan skill (angka, kata umum pendek)
            if token.isdigit():
                continue

            # Fuzzy match terhadap vocabulary
            result = process.extractOne(
                token,
                self.skill_list,
                scorer=fuzz.ratio,
                score_cutoff=self.fuzzy_threshold
            )

            if result:
                skill, score, _ = result
                matched.add(skill)

        return matched

    def _extract_skills_from_text(self, text):
        """
        Extract skills dari teks yang sudah di-clean.

        Multi-layer matching:
        1. Direct match (exact + normalisasi)
        2. Fuzzy match (typo tolerance)

        Returns:
            dict: {skills, direct_matched, fuzzy_matched}
        """
        # Clean text
        cleaned = self._clean_text(text)

        # Tokenize (words + bigrams)
        words, bigrams = self._tokenize(cleaned)
        all_tokens = words + bigrams

        # Layer 1: Direct match
        direct_matched = self._direct_match(all_tokens)

        # Layer 2: Fuzzy match (hanya untuk yang belum ketemu)
        fuzzy_matched = self._fuzzy_match(words, direct_matched)

        # Combine & deduplicate
        all_skills = direct_matched | fuzzy_matched

        return {
            'skills': sorted(list(all_skills)),
            'direct_matched': sorted(list(direct_matched)),
            'fuzzy_matched': sorted(list(fuzzy_matched)),
        }

    # =========================================================================
    # MAIN PUBLIC METHOD
    # =========================================================================

    def extract_skills_from_pdf(self, pdf_path=None, pdf_bytes=None):
        """
        Main method: Extract tech skills dari file PDF.

        Args:
            pdf_path: path ke file PDF (untuk file lokal)
            pdf_bytes: bytes content PDF (untuk upload via API)

        Returns:
            dict: {
                extracted_skills: list,
                raw_text_preview: str (first 500 chars),
                pages_processed: int,
                extraction_method: str,
                total_skills_found: int,
                direct_matched: list,
                fuzzy_matched: list,
                success: bool,
                error: str or None
            }
        """
        # Validate input
        if pdf_path is None and pdf_bytes is None:
            return {
                'extracted_skills': [],
                'raw_text_preview': '',
                'pages_processed': 0,
                'extraction_method': 'none',
                'total_skills_found': 0,
                'direct_matched': [],
                'fuzzy_matched': [],
                'success': False,
                'error': 'No PDF input provided (need pdf_path or pdf_bytes)'
            }

        if pdf_path and not os.path.exists(pdf_path):
            return {
                'extracted_skills': [],
                'raw_text_preview': '',
                'pages_processed': 0,
                'extraction_method': 'none',
                'total_skills_found': 0,
                'direct_matched': [],
                'fuzzy_matched': [],
                'success': False,
                'error': f'File not found: {pdf_path}'
            }

        # Step 1: Extract text from PDF
        extraction = self.extract_text(pdf_path=pdf_path, pdf_bytes=pdf_bytes)

        if not extraction['success']:
            return {
                'extracted_skills': [],
                'raw_text_preview': extraction['text'][:500] if extraction['text'] else '',
                'pages_processed': extraction['n_pages'],
                'extraction_method': extraction['method'],
                'total_skills_found': 0,
                'direct_matched': [],
                'fuzzy_matched': [],
                'success': False,
                'error': 'Could not extract text from PDF. File may be empty or corrupted.'
            }

        # Step 2: Extract skills from text
        skill_result = self._extract_skills_from_text(extraction['text'])

        return {
            'extracted_skills': skill_result['skills'],
            'raw_text_preview': extraction['text'][:500],
            'pages_processed': extraction['n_pages'],
            'extraction_method': extraction['method'],
            'total_skills_found': len(skill_result['skills']),
            'direct_matched': skill_result['direct_matched'],
            'fuzzy_matched': skill_result['fuzzy_matched'],
            'success': True,
            'error': None
        }

    def extract_skills_from_text_direct(self, text):
        """
        Extract skills dari teks mentah (tanpa PDF parsing).
        Berguna untuk testing atau input teks langsung.

        Args:
            text: string teks mentah

        Returns:
            dict: sama seperti extract_skills_from_pdf
        """
        if not text or len(text.strip()) == 0:
            return {
                'extracted_skills': [],
                'raw_text_preview': '',
                'pages_processed': 0,
                'extraction_method': 'direct_text',
                'total_skills_found': 0,
                'direct_matched': [],
                'fuzzy_matched': [],
                'success': False,
                'error': 'Empty text input'
            }

        skill_result = self._extract_skills_from_text(text)

        return {
            'extracted_skills': skill_result['skills'],
            'raw_text_preview': text[:500],
            'pages_processed': 0,
            'extraction_method': 'direct_text',
            'total_skills_found': len(skill_result['skills']),
            'direct_matched': skill_result['direct_matched'],
            'fuzzy_matched': skill_result['fuzzy_matched'],
            'success': True,
            'error': None
        }


# =============================================================================
# DEMO / MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("OCR PIPELINE DEMO - CV Skill Extraction")
    print("=" * 70)
    print()

    # Initialize parser
    parser = CVParser(vocab_path='data/skill_vocabulary.csv')

    # --- Demo 1: Extract from text (simulasi CV content) ---
    print("-" * 70)
    print("Demo 1: Extract Skills from Text (Simulasi CV)")
    print("-" * 70)

    sample_cv_text = """
    CURRICULUM VITAE

    Nama: Ahmad Fauzan
    Email: ahmad.fauzan@email.com
    Phone: +6281234567890

    PENDIDIKAN
    Universitas Indonesia - Teknik Informatika (2022-2026)
    IPK: 3.75

    PENGALAMAN
    - Intern Frontend Developer di PT XYZ (2025)
      Mengembangkan aplikasi web menggunakan React.js dan TypeScript.
      Implementasi REST API dengan Node.js dan Express.
      Menggunakan Git untuk kontrol versi dan CI/CD pipeline.

    - Project Freelance (2024)
      Membuat website e-commerce dengan Next.js dan Tailwind CSS.
      Database menggunakan PostgreSQL dan MongoDB.
      Deployment menggunakan Docker dan AWS.

    SKILLS
    - Programming: JavaScript, TypeScript, Python, C++
    - Frontend: React.js, Next.js, HTML, CSS, Tailwind
    - Backend: Node.js, Express.js, FastAPI
    - Database: PostgreSQL, MongoDB, Redis
    - Tools: Git, Docker, AWS, Postman
    - Lainnya: Agile, CI/CD, Linux

    SERTIFIKASI
    - AWS Cloud Practitioner
    - Google IT Support Professional
    """

    result = parser.extract_skills_from_text_direct(sample_cv_text)

    print(f"  Success: {result['success']}")
    print(f"  Method: {result['extraction_method']}")
    print(f"  Total skills found: {result['total_skills_found']}")
    print(f"  Direct matched: {result['direct_matched']}")
    print(f"  Fuzzy matched: {result['fuzzy_matched']}")
    print(f"  All extracted: {result['extracted_skills']}")
    print()

    # --- Demo 2: Extract from text (Bahasa Indonesia) ---
    print("-" * 70)
    print("Demo 2: Extract Skills dari CV Bahasa Indonesia")
    print("-" * 70)

    sample_cv_indo = """
    PENGALAMAN KERJA

    Magang di PT ABC sebagai Web Developer (2025)
    - Menguasai pengembangan web menggunakan React dan JavaScript
    - Memahami basis data relasional dan NoSQL
    - Berpengalaman dalam keamanan jaringan dan pengujian perangkat lunak
    - Familiar dengan kontrol versi menggunakan Git
    - Memahami komputasi awan dan kontainerisasi menggunakan Docker
    - Menguasai bahasa pemrograman Python dan Java
    - Berpengalaman dalam analisis data menggunakan Pandas dan NumPy
    """

    result_indo = parser.extract_skills_from_text_direct(sample_cv_indo)

    print(f"  Success: {result_indo['success']}")
    print(f"  Total skills found: {result_indo['total_skills_found']}")
    print(f"  Direct matched: {result_indo['direct_matched']}")
    print(f"  Fuzzy matched: {result_indo['fuzzy_matched']}")
    print(f"  All extracted: {result_indo['extracted_skills']}")
    print()

    # --- Demo 3: Extract from PDF (jika ada file test) ---
    print("-" * 70)
    print("Demo 3: Extract Skills from PDF")
    print("-" * 70)

    test_pdf = "tests/testing_cv.pdf"
    if os.path.exists(test_pdf):
        result_pdf = parser.extract_skills_from_pdf(pdf_path=test_pdf)
        print(f"  Success: {result_pdf['success']}")
        print(f"  Method: {result_pdf['extraction_method']}")
        print(f"  Pages: {result_pdf['pages_processed']}")
        print(f"  Skills: {result_pdf['extracted_skills']}")
    else:
        print(f"  File '{test_pdf}' tidak ditemukan.")
        print(f"  Upload file PDF untuk test, atau gunakan Demo 1/2 (text input).")
    print()

    # --- Demo 4: Edge cases ---
    print("-" * 70)
    print("Demo 4: Edge Cases")
    print("-" * 70)

    # Empty text
    result_empty = parser.extract_skills_from_text_direct("")
    print(f"  Empty text: success={result_empty['success']}, "
          f"error={result_empty['error']}")

    # Text tanpa skill
    result_no_skill = parser.extract_skills_from_text_direct(
        "Saya adalah mahasiswa yang rajin dan bertanggung jawab."
    )
    print(f"  No skills text: found={result_no_skill['total_skills_found']}, "
          f"skills={result_no_skill['extracted_skills']}")

    # Typo
    result_typo = parser.extract_skills_from_text_direct(
        "Saya menguasai Javascrpt, Pythn, dan Reakt"
    )
    print(f"  Typo text: found={result_typo['total_skills_found']}, "
          f"skills={result_typo['extracted_skills']}")

    print()
    print("=" * 70)
    print("OCR PIPELINE DEMO COMPLETE")
    print("=" * 70)
