"""
=============================================================================
PAIR GENERATION SCRIPT - Magang-in Project (v2 - Revised)
=============================================================================
Script ini menghasilkan dataset training (positive & negative pairs)
dari data lowongan magang untuk melatih Siamese Network.

Revisi dari v1:
- Normalisasi skill duplikat (ci/cd -> cicd)
- Positive pairs minimum overlap dipaksa >= 70%
- Menggunakan math.ceil agar pembulatan tidak menurunkan overlap
- Menambahkan lebih banyak variasi pair untuk dataset lebih kaya
- Statistik dan validasi lebih ketat

Cara pakai:
1. Upload file ini + magangin_jobs_cleaned.csv ke Google Colab
2. Jalankan: !python pair_generation.py
   atau copy-paste ke cell notebook

Output: file CSV 'training_pairs.csv' berisi pasangan skill + label
=============================================================================
"""

import pandas as pd
import numpy as np
from math import ceil
from itertools import combinations
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seed untuk reproducibility
random.seed(42)
np.random.seed(42)

# =============================================================================
# STEP 1: Load Dataset
# =============================================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

# Ganti path ini sesuai lokasi file di Colab
# Di Colab biasanya: '/content/magangin_jobs_cleaned.csv'
df = pd.read_csv('data/magangin_jobs_cleaned.csv')

print(f"Total lowongan: {len(df)}")
print(f"Kolom: {list(df.columns)}")
print()

# =============================================================================
# STEP 2: Extract, Clean & Normalize Skills
# =============================================================================
print("=" * 60)
print("STEP 2: Extracting, Cleaning & Normalizing Skills")
print("=" * 60)

# Mapping normalisasi skill duplikat/inkonsisten
SKILL_NORMALIZE_MAP = {
    'ci/cd': 'cicd',
    'ci cd': 'cicd',
    'c++': 'cpp',
    'c#': 'csharp',
    # Tambahkan mapping lain jika ditemukan duplikat di masa depan
    # 'react.js': 'react',
    # 'node.js': 'node',
}


def parse_and_normalize_skills(skill_string):
    """Mengubah string skills menjadi list yang sudah dinormalisasi."""
    if pd.isna(skill_string) or skill_string == '':
        return []
    skills = [s.strip().lower() for s in str(skill_string).split(',') if s.strip()]
    # Normalisasi
    normalized = []
    for skill in skills:
        skill = SKILL_NORMALIZE_MAP.get(skill, skill)
        if skill not in normalized:  # Hindari duplikat setelah normalisasi
            normalized.append(skill)
    return normalized


df['skills_list'] = df['skills'].apply(parse_and_normalize_skills)

# Filter job yang punya minimal 2 skills
df_valid = df[df['skills_list'].apply(len) >= 2].reset_index(drop=True)
print(f"Lowongan total: {len(df)}")
print(f"Lowongan dengan >= 2 skills (dipakai): {len(df_valid)}")
print(f"Lowongan dengan < 2 skills (dibuang dari training): {len(df) - len(df_valid)}")

# Kumpulkan semua unique skills
all_skills = set()
for skills in df_valid['skills_list']:
    all_skills.update(skills)

all_skills = sorted(list(all_skills))
print(f"\nTotal unique skills (setelah normalisasi): {len(all_skills)}")
print(f"Daftar skills: {all_skills}")
print()

# Mapping role untuk negative pair generation
print("Roles yang tersedia:")
print(df_valid['role'].value_counts())
print()

# =============================================================================
# STEP 3: Generate Positive Pairs (label = 1)
# Aturan: overlap HARUS >= 70% terhadap job skills
# =============================================================================
print("=" * 60)
print("STEP 3: Generating Positive Pairs (label = 1, overlap >= 70%)")
print("=" * 60)

positive_pairs = []

MIN_POSITIVE_OVERLAP = 0.70  # Threshold minimum untuk positive pair

for idx, row in df_valid.iterrows():
    job_skills = row['skills_list']
    job_title = row['title']
    n_skills = len(job_skills)

    if n_skills < 2:
        continue

    # --- Type A: Exact match (100%) ---
    positive_pairs.append({
        'user_skills': job_skills.copy(),
        'job_skills': job_skills.copy(),
        'job_title': job_title,
        'job_idx': idx,
        'overlap_pct': 1.0,
        'label': 1
    })

    # --- Type B: High match (70-95% subset of job skills) ---
    # Gunakan ceil agar tidak turun di bawah 70%
    n_samples_b = min(5, max(2, n_skills))

    for _ in range(n_samples_b):
        # Hitung minimum skill yang harus diambil agar overlap >= 70%
        min_select = ceil(n_skills * MIN_POSITIVE_OVERLAP)
        max_select = n_skills - 1 if n_skills > 2 else n_skills

        if min_select > max_select:
            min_select = max_select

        n_select = random.randint(min_select, max_select)
        user_skills = random.sample(job_skills, n_select)
        overlap = len(set(user_skills) & set(job_skills)) / len(job_skills)

        # Safety check: pastikan >= 70%
        if overlap >= MIN_POSITIVE_OVERLAP:
            positive_pairs.append({
                'user_skills': user_skills,
                'job_skills': job_skills.copy(),
                'job_title': job_title,
                'job_idx': idx,
                'overlap_pct': overlap,
                'label': 1
            })

    # --- Type C: Over-qualified user (punya semua/hampir semua skill job + extra) ---
    n_samples_c = 3

    for _ in range(n_samples_c):
        # Ambil 80-100% skill job
        min_select = ceil(n_skills * 0.80)
        n_select = random.randint(min_select, n_skills)
        user_skills = random.sample(job_skills, n_select)

        # Tambah 1-4 skill random yang tidak ada di job
        other_skills = [s for s in all_skills if s not in job_skills]
        if other_skills:
            n_extra = random.randint(1, min(4, len(other_skills)))
            extra = random.sample(other_skills, n_extra)
            user_skills.extend(extra)

        overlap = len(set(user_skills) & set(job_skills)) / len(job_skills)

        if overlap >= MIN_POSITIVE_OVERLAP:
            positive_pairs.append({
                'user_skills': user_skills,
                'job_skills': job_skills.copy(),
                'job_title': job_title,
                'job_idx': idx,
                'overlap_pct': overlap,
                'label': 1
            })

    # --- Type D: Exact match tapi urutan beda (untuk variasi) ---
    shuffled = job_skills.copy()
    random.shuffle(shuffled)
    positive_pairs.append({
        'user_skills': shuffled,
        'job_skills': job_skills.copy(),
        'job_title': job_title,
        'job_idx': idx,
        'overlap_pct': 1.0,
        'label': 1
    })

# Validasi: buang pair yang overlap < 70% (safety net)
positive_pairs = [p for p in positive_pairs if p['overlap_pct'] >= MIN_POSITIVE_OVERLAP]

print(f"Total positive pairs: {len(positive_pairs)}")
if positive_pairs:
    overlaps = [p['overlap_pct'] for p in positive_pairs]
    print(f"  Overlap min: {min(overlaps):.2%}")
    print(f"  Overlap max: {max(overlaps):.2%}")
    print(f"  Overlap mean: {np.mean(overlaps):.2%}")
print()

# =============================================================================
# STEP 4: Generate Negative Pairs (label = 0)
# Aturan: overlap HARUS < 30% terhadap job skills
# =============================================================================
print("=" * 60)
print("STEP 4: Generating Negative Pairs (label = 0, overlap < 30%)")
print("=" * 60)

negative_pairs = []

MAX_NEGATIVE_OVERLAP = 0.30  # Threshold maximum untuk negative pair

# Group jobs by role untuk cross-domain sampling
role_groups = df_valid.groupby('role')['skills_list'].apply(list).to_dict()
role_names = list(role_groups.keys())

# --- Strategy 1: Cross-domain pairs (user dari role berbeda) ---
for idx, row in df_valid.iterrows():
    job_skills = row['skills_list']
    job_role = row['role']
    job_title = row['title']
    n_skills = len(job_skills)

    other_roles = [r for r in role_names if r != job_role]

    for _ in range(4):  # 4 attempts per job
        if not other_roles:
            break

        other_role = random.choice(other_roles)
        other_job_skills_list = role_groups[other_role]

        if not other_job_skills_list:
            continue

        other_job_skills = random.choice(other_job_skills_list)

        overlap = (len(set(other_job_skills) & set(job_skills)) / len(job_skills)
                   if len(job_skills) > 0 else 0)

        if overlap < MAX_NEGATIVE_OVERLAP:
            negative_pairs.append({
                'user_skills': other_job_skills.copy(),
                'job_skills': job_skills.copy(),
                'job_title': job_title,
                'job_idx': idx,
                'overlap_pct': overlap,
                'label': 0
            })

# --- Strategy 2: Random skill set yang tidak ada di job ---
for idx, row in df_valid.iterrows():
    job_skills = row['skills_list']
    job_title = row['title']

    for _ in range(3):  # 3 samples per job
        non_job_skills = [s for s in all_skills if s not in job_skills]

        if len(non_job_skills) < 2:
            continue

        n_random = random.randint(2, min(6, len(non_job_skills)))
        user_skills = random.sample(non_job_skills, n_random)

        overlap = len(set(user_skills) & set(job_skills)) / len(job_skills)

        if overlap < MAX_NEGATIVE_OVERLAP:
            negative_pairs.append({
                'user_skills': user_skills,
                'job_skills': job_skills.copy(),
                'job_title': job_title,
                'job_idx': idx,
                'overlap_pct': overlap,
                'label': 0
            })

# --- Strategy 3: Hard negatives (1-2 skill overlap, tapi tetap < 30%) ---
for idx, row in df_valid.iterrows():
    job_skills = row['skills_list']
    job_title = row['title']
    n_skills = len(job_skills)

    if n_skills < 4:  # Hanya untuk job dengan cukup skill
        continue

    for _ in range(2):
        # Ambil 1 skill dari job
        n_match = random.randint(1, max(1, int(n_skills * 0.25)))
        matching_skills = random.sample(job_skills, n_match)

        # Tambah skill yang tidak ada di job
        non_job_skills = [s for s in all_skills if s not in job_skills]
        if len(non_job_skills) < 2:
            continue

        n_extra = random.randint(2, min(5, len(non_job_skills)))
        extra_skills = random.sample(non_job_skills, n_extra)

        user_skills = matching_skills + extra_skills
        overlap = len(set(user_skills) & set(job_skills)) / len(job_skills)

        if overlap < MAX_NEGATIVE_OVERLAP:
            negative_pairs.append({
                'user_skills': user_skills,
                'job_skills': job_skills.copy(),
                'job_title': job_title,
                'job_idx': idx,
                'overlap_pct': overlap,
                'label': 0
            })

# Validasi: buang pair yang overlap >= 30% (safety net)
negative_pairs = [p for p in negative_pairs if p['overlap_pct'] < MAX_NEGATIVE_OVERLAP]

print(f"Total negative pairs: {len(negative_pairs)}")
if negative_pairs:
    overlaps = [p['overlap_pct'] for p in negative_pairs]
    print(f"  Overlap min: {min(overlaps):.2%}")
    print(f"  Overlap max: {max(overlaps):.2%}")
    print(f"  Overlap mean: {np.mean(overlaps):.2%}")
print()

# =============================================================================
# STEP 5: Balance Dataset & Combine
# =============================================================================
print("=" * 60)
print("STEP 5: Balancing & Combining Dataset")
print("=" * 60)

n_positive = len(positive_pairs)
n_negative = len(negative_pairs)

print(f"Before balancing:")
print(f"  Positive: {n_positive}")
print(f"  Negative: {n_negative}")

# Undersample yang lebih banyak
if n_positive > n_negative:
    positive_pairs = random.sample(positive_pairs, n_negative)
elif n_negative > n_positive:
    negative_pairs = random.sample(negative_pairs, n_positive)

print(f"\nAfter balancing:")
print(f"  Positive: {len(positive_pairs)}")
print(f"  Negative: {len(negative_pairs)}")

# Combine & shuffle
all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)

print(f"\nTotal training pairs: {len(all_pairs)}")
print()

# =============================================================================
# STEP 6: Convert to DataFrame & Save
# =============================================================================
print("=" * 60)
print("STEP 6: Saving to CSV")
print("=" * 60)

df_pairs = pd.DataFrame(all_pairs)
df_pairs['user_skills_str'] = df_pairs['user_skills'].apply(lambda x: ', '.join(sorted(x)))
df_pairs['job_skills_str'] = df_pairs['job_skills'].apply(lambda x: ', '.join(sorted(x)))
df_pairs['n_user_skills'] = df_pairs['user_skills'].apply(len)
df_pairs['n_job_skills'] = df_pairs['job_skills'].apply(len)

# Save
output_path = 'data/training_pairs.csv'
df_pairs[['user_skills_str', 'job_skills_str', 'job_title', 'job_idx',
           'n_user_skills', 'n_job_skills', 'overlap_pct', 'label']].to_csv(output_path, index=False)

print(f"Saved to: {output_path}")
print(f"Total rows: {len(df_pairs)}")
print()

# =============================================================================
# STEP 7: Dataset Statistics & Validation
# =============================================================================
print("=" * 60)
print("STEP 7: Dataset Statistics & Validation")
print("=" * 60)

print(f"\nLabel distribution:")
print(df_pairs['label'].value_counts())

print(f"\nOverlap percentage stats (Positive pairs, label=1):")
pos_df = df_pairs[df_pairs['label'] == 1]
print(f"  Count: {len(pos_df)}")
print(f"  Mean:  {pos_df['overlap_pct'].mean():.2%}")
print(f"  Min:   {pos_df['overlap_pct'].min():.2%}")
print(f"  Max:   {pos_df['overlap_pct'].max():.2%}")

print(f"\nOverlap percentage stats (Negative pairs, label=0):")
neg_df = df_pairs[df_pairs['label'] == 0]
print(f"  Count: {len(neg_df)}")
print(f"  Mean:  {neg_df['overlap_pct'].mean():.2%}")
print(f"  Min:   {neg_df['overlap_pct'].min():.2%}")
print(f"  Max:   {neg_df['overlap_pct'].max():.2%}")

print(f"\nUser skills count stats:")
print(f"  Mean: {df_pairs['n_user_skills'].mean():.1f}")
print(f"  Min:  {df_pairs['n_user_skills'].min()}")
print(f"  Max:  {df_pairs['n_user_skills'].max()}")

print(f"\nJob skills count stats:")
print(f"  Mean: {df_pairs['n_job_skills'].mean():.1f}")
print(f"  Min:  {df_pairs['n_job_skills'].min()}")
print(f"  Max:  {df_pairs['n_job_skills'].max()}")

# Validation checks
print(f"\n{'=' * 60}")
print("VALIDATION CHECKS")
print(f"{'=' * 60}")

violations_pos = len(pos_df[pos_df['overlap_pct'] < MIN_POSITIVE_OVERLAP])
violations_neg = len(neg_df[neg_df['overlap_pct'] >= MAX_NEGATIVE_OVERLAP])

print(f"  Positive pairs with overlap < 70%: {violations_pos} {'[PASS]' if violations_pos == 0 else '[FAIL]'}")
print(f"  Negative pairs with overlap >= 30%: {violations_neg} {'[PASS]' if violations_neg == 0 else '[FAIL]'}")
print(f"  Dataset balanced: {'[PASS]' if len(pos_df) == len(neg_df) else '[FAIL]'}")
print(f"  Total pairs > 1000: {'[PASS]' if len(df_pairs) > 1000 else '[WARNING - consider adding more data]'}")

# Grey zone info
print(f"\n  Grey zone (30%-70%) is excluded from training.")
print(f"  This helps the model learn clear boundaries between 'match' and 'no match'.")

print(f"\n{'=' * 60}")
print("DONE! File 'training_pairs.csv' siap digunakan untuk training.")
print(f"{'=' * 60}")

# =============================================================================
# STEP 8: Preview Data
# =============================================================================
print("\n\nPreview 10 data pertama:")
print("-" * 60)
for i, row in df_pairs.head(10).iterrows():
    label_text = "COCOK" if row['label'] == 1 else "TIDAK COCOK"
    print(f"\n[{label_text}] Overlap: {row['overlap_pct']:.0%}")
    print(f"  User skills: {row['user_skills_str']}")
    print(f"  Job skills:  {row['job_skills_str']}")
    print(f"  Job: {row['job_title']}")

# =============================================================================
# STEP 9: Save skill vocabulary (untuk dipakai saat training & inference)
# =============================================================================
print("\n\n" + "=" * 60)
print("STEP 9: Saving Skill Vocabulary")
print("=" * 60)

# Simpan daftar semua unique skills sebagai referensi
skill_vocab_path = 'data/skill_vocabulary.csv'
skill_vocab_df = pd.DataFrame({'skill': all_skills, 'index': range(len(all_skills))})
skill_vocab_df.to_csv(skill_vocab_path, index=False)
print(f"Saved skill vocabulary to: {skill_vocab_path}")
print(f"Total skills in vocabulary: {len(all_skills)}")
print("\nVocabulary ini akan dipakai untuk:")
print("  - Multi-hot encoding saat training")
print("  - Encoding input user saat inference")
print("  - Fuzzy matching saat OCR CV parsing")
