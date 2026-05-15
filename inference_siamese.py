"""
=============================================================================
INFERENCE SCRIPT - Siamese Network for Skill Matching (v3 - Final)
=============================================================================
Magang-in Project - AI Team

Script ini memenuhi:
- Main Quest 4: Membuat kode sederhana untuk proses inference model

Keputusan Desain:
- Formula: final_score = 0.4 * model_score + 0.6 * coverage_score
- Match Category: Strong (>=60%), Partial (30-59%), Low (<30%)
- Filter: minimal 1 skill match (bukan persen)
- Output: full detail per result (tanpa display message, frontend handle)
- Ranking: berdasarkan final_score (hybrid)

Cara pakai di Google Colab:
1. Upload file ini + siamese_model.keras + skill_vocabulary.csv
   + model_threshold.txt + magangin_jobs_cleaned.csv
2. Jalankan: !python inference_siamese.py

Atau import sebagai module:
    from inference_siamese import SkillMatcher
    matcher = SkillMatcher()
    results = matcher.rank_jobs(["react", "javascript", "node"])
=============================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1: CUSTOM OBJECTS (untuk load model)
# =============================================================================

@keras.utils.register_keras_serializable(package="MaganginAI")
class DistanceLayer(keras.layers.Layer):
    """Custom Layer - harus didefinisikan ulang untuk load model."""

    def __init__(self, **kwargs):
        super(DistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        user_embedding, job_embedding = inputs
        return tf.abs(user_embedding - job_embedding)

    def get_config(self):
        return super(DistanceLayer, self).get_config()


@keras.utils.register_keras_serializable(package="MaganginAI")
class ContrastiveLoss(keras.losses.Loss):
    """Custom Loss - harus didefinisikan ulang untuk load model."""

    def __init__(self, margin=1.0, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        distance = 1.0 - y_pred
        positive_loss = y_true * tf.square(distance)
        negative_loss = (1.0 - y_true) * tf.square(
            tf.maximum(self.margin - distance, 0.0)
        )
        return tf.reduce_mean(positive_loss + negative_loss)

    def get_config(self):
        config = super(ContrastiveLoss, self).get_config()
        config.update({"margin": self.margin})
        return config


# =============================================================================
# SECTION 2: SKILL MATCHER CLASS
# =============================================================================

class SkillMatcher:
    """
    Kelas utama untuk inference skill matching dengan hybrid scoring.

    Scoring:
        model_score    = output Siamese Network (semantic similarity)
        coverage_score = matched_skills / total_job_skills
        final_score    = 0.4 * model_score + 0.6 * coverage_score

    Match Category (berdasarkan coverage_score):
        Strong Match  : coverage >= 60%
        Partial Match : coverage 30-59%
        Low Match     : coverage < 30%

    Filter:
        Default: minimal 1 skill match (job tanpa skill match tidak ditampilkan)
    """

    # --- Scoring weights ---
    MODEL_WEIGHT = 0.4
    COVERAGE_WEIGHT = 0.6

    # --- Match category thresholds (berdasarkan coverage) ---
    STRONG_MATCH_THRESHOLD = 0.60
    PARTIAL_MATCH_THRESHOLD = 0.30

    # --- Skill normalization mapping ---
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
        'mongo': 'mongodb',
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

    def __init__(self,
                 model_path='model/siamese_model.keras',
                 vocab_path='data/skill_vocabulary.csv',
                 threshold_path='model/model_threshold.txt',
                 jobs_path='data/magangin_jobs_cleaned.csv'):
        """Load model, vocabulary, threshold, dan data lowongan."""
        print("Loading Skill Matcher...")

        # Load model
        self.model = keras.models.load_model(model_path)
        print(f"  Model loaded: {model_path}")

        # Load vocabulary
        vocab_df = pd.read_csv(vocab_path)
        self.skill_list = vocab_df['skill'].tolist()
        self.skill_to_idx = {skill: idx for idx, skill in enumerate(self.skill_list)}
        self.vocab_size = len(self.skill_list)
        print(f"  Vocabulary loaded: {self.vocab_size} skills")

        # Load threshold (dari training, untuk referensi)
        with open(threshold_path, 'r') as f:
            self.model_threshold = float(f.read().strip())
        print(f"  Model threshold: {self.model_threshold:.2f}")

        # Load jobs data
        self.jobs_df = pd.read_csv(jobs_path)
        self.jobs_df['skills_list'] = self.jobs_df['skills'].apply(
            self._parse_skills
        )
        # Filter job dengan minimal 1 skill
        self.jobs_df = self.jobs_df[
            self.jobs_df['skills_list'].apply(len) > 0
        ].reset_index(drop=True)
        print(f"  Jobs loaded: {len(self.jobs_df)} lowongan")

        # Pre-encode semua job skills
        self.jobs_encoded = np.array([
            self._encode_skills(row['skills_list'])
            for _, row in self.jobs_df.iterrows()
        ])
        print(f"  Jobs pre-encoded: shape {self.jobs_encoded.shape}")

        print(f"\n  Scoring config:")
        print(f"    Formula: {self.MODEL_WEIGHT}*model + {self.COVERAGE_WEIGHT}*coverage")
        print(f"    Strong Match: coverage >= {self.STRONG_MATCH_THRESHOLD*100:.0f}%")
        print(f"    Partial Match: coverage >= {self.PARTIAL_MATCH_THRESHOLD*100:.0f}%")
        print(f"    Filter: min 1 skill match")
        print("Skill Matcher ready!\n")

    # --- Private Methods ---

    def _parse_skills(self, skill_string):
        """Parse dan normalisasi skill string."""
        if pd.isna(skill_string) or skill_string == '':
            return []
        skills = [s.strip().lower() for s in str(skill_string).split(',')]
        normalized = []
        for skill in skills:
            skill = self.SKILL_NORMALIZE_MAP.get(skill, skill)
            if skill not in normalized:
                normalized.append(skill)
        return normalized

    def _encode_skills(self, skills_list):
        """Encode list of skills ke multi-hot vector."""
        vector = np.zeros(self.vocab_size, dtype=np.float32)
        for skill in skills_list:
            skill_normalized = self.SKILL_NORMALIZE_MAP.get(skill, skill)
            if skill_normalized in self.skill_to_idx:
                vector[self.skill_to_idx[skill_normalized]] = 1.0
        return vector

    def _categorize_match(self, coverage_score):
        """Kategorikan match berdasarkan coverage score."""
        if coverage_score >= self.STRONG_MATCH_THRESHOLD:
            return "Strong Match"
        elif coverage_score >= self.PARTIAL_MATCH_THRESHOLD:
            return "Partial Match"
        else:
            return "Low Match"

    def _compute_final_score(self, model_score, coverage_score):
        """Hitung hybrid final score."""
        return (self.MODEL_WEIGHT * model_score +
                self.COVERAGE_WEIGHT * coverage_score)

    # --- Public Methods ---

    def normalize_user_skills(self, user_skills):
        """
        Normalisasi input skill user.

        Args:
            user_skills: list of strings (raw input dari user/OCR)

        Returns:
            tuple: (normalized_skills, unknown_skills)
        """
        normalized = []
        unknown = []

        for skill in user_skills:
            skill_lower = skill.strip().lower()
            skill_mapped = self.SKILL_NORMALIZE_MAP.get(skill_lower, skill_lower)

            if skill_mapped in self.skill_to_idx:
                if skill_mapped not in normalized:
                    normalized.append(skill_mapped)
            else:
                unknown.append(skill)

        return normalized, unknown

    def predict_match(self, user_skills, job_skills):
        """
        Prediksi kecocokan antara skill user dan skill job.

        Args:
            user_skills: list of strings
            job_skills: list of strings

        Returns:
            dict: model_score, coverage_score, final_score, category, dll
        """
        # Normalize
        user_normalized, _ = self.normalize_user_skills(user_skills)
        job_normalized, _ = self.normalize_user_skills(job_skills)

        if not user_normalized or not job_normalized:
            return {
                'model_score': 0.0,
                'coverage_score': 0.0,
                'final_score': 0.0,
                'match_category': 'Low Match',
                'matched_skills': [],
                'missing_skills': list(job_normalized),
                'n_matched': 0,
                'n_required': len(job_normalized),
            }

        # Encode
        user_vector = self._encode_skills(user_normalized).reshape(1, -1)
        job_vector = self._encode_skills(job_normalized).reshape(1, -1)

        # Model score (semantic similarity)
        model_score = float(self.model.predict(
            [user_vector, job_vector], verbose=0
        ).flatten()[0])

        # Coverage score
        user_set = set(user_normalized)
        job_set = set(job_normalized)
        matched = sorted(list(user_set & job_set))
        missing = sorted(list(job_set - user_set))
        coverage_score = len(matched) / len(job_set) if job_set else 0.0

        # Final hybrid score
        final_score = self._compute_final_score(model_score, coverage_score)

        # Category
        category = self._categorize_match(coverage_score)

        return {
            'model_score': model_score,
            'coverage_score': coverage_score,
            'final_score': final_score,
            'match_category': category,
            'matched_skills': matched,
            'missing_skills': missing,
            'n_matched': len(matched),
            'n_required': len(job_set),
        }

    def rank_jobs(self, user_skills, top_n=10, min_matched=1,
                  sort_by='final_score'):
        """
        Ranking lowongan berdasarkan kecocokan dengan skill user.

        Args:
            user_skills: list of strings (raw input)
            top_n: jumlah rekomendasi teratas
            min_matched: minimal jumlah skill yang harus match (default: 1)
            sort_by: 'final_score' | 'model_score' | 'coverage_score'

        Returns:
            list of dict: ranking lowongan dengan hybrid scoring
        """
        # Normalisasi input
        normalized_skills, unknown_skills = self.normalize_user_skills(user_skills)

        if unknown_skills:
            print(f"  Warning: Skill tidak dikenali (diabaikan): {unknown_skills}")

        if not normalized_skills:
            print("  Error: Tidak ada skill valid yang dikenali.")
            return []

        # Encode user skills
        user_vector = self._encode_skills(normalized_skills)
        user_batch = np.tile(user_vector, (len(self.jobs_df), 1))

        # Batch inference (predict semua job sekaligus)
        model_scores = self.model.predict(
            [user_batch, self.jobs_encoded], verbose=0
        ).flatten()

        # Build results
        user_skills_set = set(normalized_skills)
        results = []

        for i, (_, job_row) in enumerate(self.jobs_df.iterrows()):
            job_skills_set = set(job_row['skills_list'])

            if not job_skills_set:
                continue

            # Gap analysis
            matched_skills = sorted(list(job_skills_set & user_skills_set))
            missing_skills = sorted(list(job_skills_set - user_skills_set))

            # Filter: minimal N skill match
            if len(matched_skills) < min_matched:
                continue

            # Scores
            model_score = float(model_scores[i])
            coverage_score = len(matched_skills) / len(job_skills_set)
            final_score = self._compute_final_score(model_score, coverage_score)

            # Category
            category = self._categorize_match(coverage_score)

            results.append({
                'rank': 0,
                'title': job_row['title'],
                'company': job_row['company_name'],
                'model_score': model_score,
                'coverage_score': coverage_score,
                'final_score': final_score,
                'match_category': category,
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'n_matched': len(matched_skills),
                'n_required': len(job_skills_set),
                'location': job_row.get('location_city', 'Unknown'),
                'role': job_row.get('role', 'Unknown'),
                'roadmap_url': job_row.get('roadmap_url', ''),
            })

        # Sort
        if sort_by not in ('final_score', 'model_score', 'coverage_score'):
            sort_by = 'final_score'
        results.sort(key=lambda x: x[sort_by], reverse=True)

        # Assign rank
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return results[:top_n]

    def get_skill_vocabulary(self):
        """Return daftar skill yang dikenali sistem."""
        return self.skill_list.copy()

    def get_roadmap_suggestion(self, user_skills, job_skills):
        """
        Gap analysis + roadmap suggestion.

        Args:
            user_skills: list skill user (normalized)
            job_skills: list skill job (normalized)

        Returns:
            dict: matched, missing, completion_pct, suggestion text
        """
        user_set = set(user_skills)
        job_set = set(job_skills)
        matched = sorted(list(user_set & job_set))
        missing = sorted(list(job_set - user_set))

        completion = len(matched) / len(job_set) * 100 if job_set else 0

        return {
            'matched_skills': matched,
            'missing_skills': missing,
            'total_required': len(job_set),
            'total_matched': len(matched),
            'completion_pct': completion,
        }


# =============================================================================
# SECTION 3: DEMO / MAIN EXECUTION
# =============================================================================

def print_ranking(results, show_details=True):
    """Helper untuk print ranking."""
    if not results:
        print("  (Tidak ada hasil yang memenuhi filter)")
        return

    for r in results:
        cat = r['match_category']
        if cat == "Strong Match":
            icon = "[STRONG ]"
        elif cat == "Partial Match":
            icon = "[PARTIAL]"
        else:
            icon = "[LOW    ]"

        cov_pct = r['coverage_score'] * 100

        print(f"  #{r['rank']} {icon} Final: {r['final_score']:.3f} | "
              f"Model: {r['model_score']:.3f} | "
              f"Coverage: {cov_pct:.0f}% ({r['n_matched']}/{r['n_required']})")
        print(f"      {r['title']} @ {r['company']}")
        if show_details:
            print(f"      Matched:  {r['matched_skills']}")
            print(f"      Missing:  {r['missing_skills']}")
            print(f"      Location: {r['location']} | Role: {r['role']}")
        print()


if __name__ == "__main__":
    print("=" * 70)
    print("INFERENCE DEMO - Siamese Skill Matcher (v3 - Hybrid Scoring)")
    print("=" * 70)
    print()

    # Initialize
    matcher = SkillMatcher()

    # =========================================================================
    # Demo 1: Single Pair Prediction
    # =========================================================================
    print("-" * 70)
    print("Demo 1: Single Pair Prediction")
    print("-" * 70)

    user = ["react", "javascript", "typescript", "css", "html"]
    job = ["react", "javascript", "typescript", "css", "html", "nextjs", "git"]

    result = matcher.predict_match(user, job)
    print(f"User skills: {user}")
    print(f"Job skills:  {job}")
    print(f"  Model score:    {result['model_score']:.4f} (semantic similarity)")
    print(f"  Coverage score: {result['coverage_score']:.4f} "
          f"({result['n_matched']}/{result['n_required']} skills matched)")
    print(f"  Final score:    {result['final_score']:.4f} "
          f"(0.4*model + 0.6*coverage)")
    print(f"  Category:       {result['match_category']}")
    print(f"  Matched:        {result['matched_skills']}")
    print(f"  Missing:        {result['missing_skills']}")
    print()

    # =========================================================================
    # Demo 2: Ranking (User Data/AI — kasus problematic sebelumnya)
    # =========================================================================
    print("-" * 70)
    print("Demo 2: Ranking Top 5 (User: Data/AI Skills)")
    print("  Kasus yang sebelumnya bermasalah — Senior ML Engineer")
    print("  seharusnya tidak lagi di ranking teratas")
    print("-" * 70)

    user_skills = ["python", "sql", "pandas", "numpy", "tensorflow"]
    print(f"User skills: {user_skills}")
    print()

    results = matcher.rank_jobs(user_skills, top_n=5)
    print_ranking(results)

    # =========================================================================
    # Demo 3: Perbandingan Model Score vs Final Score
    # =========================================================================
    print("-" * 70)
    print("Demo 3: Perbandingan Ranking")
    print("  [A] Model score only (cara lama)")
    print("  [B] Final score / hybrid (cara baru)")
    print("-" * 70)

    print(f"\nUser skills: {user_skills}")

    print("\n[A] Sorted by MODEL SCORE only:")
    results_a = matcher.rank_jobs(user_skills, top_n=5, sort_by='model_score')
    print_ranking(results_a, show_details=False)

    print("[B] Sorted by FINAL SCORE (hybrid):")
    results_b = matcher.rank_jobs(user_skills, top_n=5, sort_by='final_score')
    print_ranking(results_b, show_details=False)

    # =========================================================================
    # Demo 4: User Frontend Skills
    # =========================================================================
    print("-" * 70)
    print("Demo 4: Ranking Top 5 (User: Frontend Skills)")
    print("-" * 70)

    user_skills_fe = ["react", "javascript", "typescript", "css", "html", "git"]
    print(f"User skills: {user_skills_fe}")
    print()

    results_fe = matcher.rank_jobs(user_skills_fe, top_n=5)
    print_ranking(results_fe)

    # =========================================================================
    # Demo 5: User dengan Skill Sedikit (Fresh Graduate)
    # =========================================================================
    print("-" * 70)
    print("Demo 5: Ranking (User: Fresh Graduate, hanya 2 skill)")
    print("  Test apakah sistem masih bisa rekomendasi untuk user skill sedikit")
    print("-" * 70)

    user_skills_fresh = ["html", "css"]
    print(f"User skills: {user_skills_fresh}")
    print()

    results_fresh = matcher.rank_jobs(user_skills_fresh, top_n=5)
    print_ranking(results_fresh)

    # =========================================================================
    # Demo 6: Skill Normalization
    # =========================================================================
    print("-" * 70)
    print("Demo 6: Skill Normalization")
    print("-" * 70)

    raw_skills = ["React.js", "Node.js", "C++", "PostgreSQL", "CI/CD",
                  "Vue.js", "Next.js", "UnknownSkill123"]
    normalized, unknown = matcher.normalize_user_skills(raw_skills)
    print(f"Input (raw):    {raw_skills}")
    print(f"Normalized:     {normalized}")
    print(f"Not recognized: {unknown}")
    print()

    # =========================================================================
    # Demo 7: Roadmap Suggestion
    # =========================================================================
    print("-" * 70)
    print("Demo 7: Roadmap Suggestion (Gap Analysis)")
    print("-" * 70)

    user_demo = ["react", "javascript", "css"]
    job_demo = ["react", "javascript", "css", "typescript", "nextjs",
                "git", "docker"]

    roadmap = matcher.get_roadmap_suggestion(user_demo, job_demo)
    print(f"User skills: {user_demo}")
    print(f"Job skills:  {job_demo}")
    print(f"  Completion:  {roadmap['completion_pct']:.0f}%")
    print(f"  Matched:     {roadmap['matched_skills']}")
    print(f"  Missing:     {roadmap['missing_skills']}")
    print(f"  Category:    {matcher._categorize_match(roadmap['total_matched']/roadmap['total_required'])}")
    print()

    # =========================================================================
    # Demo 8: Available Skills
    # =========================================================================
    print("-" * 70)
    print("Demo 8: Available Skills in System")
    print("-" * 70)
    vocab = matcher.get_skill_vocabulary()
    print(f"Total skills recognized: {len(vocab)}")
    print(f"Skills: {vocab}")

    print()
    print("=" * 70)
    print("INFERENCE DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Scoring formula: final = 0.4*model_score + 0.6*coverage_score")
    print("Match categories: Strong (>=60%), Partial (30-59%), Low (<30%)")
    print("Filter: min 1 skill match")
    print("Display message: handled by frontend (not included in API response)")
