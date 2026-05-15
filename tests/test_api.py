"""
=============================================================================
TEST API - Magang-in Skill Matching API
=============================================================================
Script untuk testing semua endpoint FastAPI setelah deployment.

Cara pakai:
  1. Pastikan server sudah berjalan (uvicorn atau docker)
  2. Jalankan: python tests/test_api.py
  3. Atau dengan custom URL: python tests/test_api.py http://api.domainmu.com

Output: status setiap endpoint (PASS/FAIL)
=============================================================================
"""

import sys
import os
import json

# Tambah parent directory ke path agar bisa import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
    USE_REQUESTS = True
except ImportError:
    USE_REQUESTS = False

from fastapi.testclient import TestClient


def get_client(base_url=None):
    """
    Buat client untuk testing.
    Jika base_url diberikan, pakai requests (test server yang sudah jalan).
    Jika tidak, pakai TestClient (test langsung tanpa server).
    """
    if base_url and USE_REQUESTS:
        return None, base_url
    else:
        # Import dan setup TestClient
        from app import app
        from inference_siamese import SkillMatcher
        from ocr_pipeline import CVParser
        import app as app_module

        app_module.matcher = SkillMatcher()
        app_module.cv_parser = CVParser()

        client = TestClient(app)
        return client, None


def make_request(client, base_url, method, endpoint, **kwargs):
    """Unified request function untuk TestClient atau requests."""
    if client:
        # TestClient
        if method == 'GET':
            return client.get(endpoint, **kwargs)
        elif method == 'POST':
            return client.post(endpoint, **kwargs)
    else:
        # requests ke server
        url = f"{base_url}{endpoint}"
        if method == 'GET':
            return requests.get(url, **kwargs)
        elif method == 'POST':
            return requests.post(url, **kwargs)


def test_health(client, base_url):
    """Test GET /api/health"""
    response = make_request(client, base_url, 'GET', '/api/health')
    data = response.json()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data['status'] == 'ok'
    assert data['model_loaded'] == True
    assert data['vocab_size'] == 80
    assert data['jobs_count'] > 0

    return f"200 OK | vocab={data['vocab_size']} | jobs={data['jobs_count']}"


def test_skills(client, base_url):
    """Test GET /api/skills"""
    response = make_request(client, base_url, 'GET', '/api/skills')
    data = response.json()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data['total'] == 80
    assert len(data['skills']) == 80
    assert 'react' in data['skills']
    assert 'python' in data['skills']

    return f"200 OK | {data['total']} skills"


def test_match(client, base_url):
    """Test POST /api/match"""
    payload = {
        "skills": ["react", "javascript", "typescript", "css", "html"],
        "top_n": 5
    }
    response = make_request(client, base_url, 'POST', '/api/match', json=payload)
    data = response.json()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert len(data['results']) <= 5
    assert len(data['results']) > 0
    assert data['unknown_skills'] == []
    assert len(data['user_skills_normalized']) == 5

    # Cek struktur result
    first = data['results'][0]
    assert 'rank' in first
    assert 'title' in first
    assert 'final_score' in first
    assert 'model_score' in first
    assert 'coverage_score' in first
    assert 'match_category' in first
    assert 'matched_skills' in first
    assert 'missing_skills' in first
    assert 'roadmap_url' in first

    top_category = first['match_category']
    top_score = first['final_score']

    return f"200 OK | {len(data['results'])} results | top: {top_score:.3f} ({top_category})"


def test_match_unknown_skills(client, base_url):
    """Test POST /api/match dengan skill yang tidak dikenali"""
    payload = {
        "skills": ["react", "UnknownSkill123", "FakeFramework"],
        "top_n": 3
    }
    response = make_request(client, base_url, 'POST', '/api/match', json=payload)
    data = response.json()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert 'UnknownSkill123' in data['unknown_skills']
    assert 'FakeFramework' in data['unknown_skills']
    assert 'react' in data['user_skills_normalized']

    return f"200 OK | unknown={data['unknown_skills']}"


def test_match_empty_skills(client, base_url):
    """Test POST /api/match dengan skill kosong (harus error)"""
    payload = {"skills": ["TotallyFakeSkill", "AnotherFake"]}
    response = make_request(client, base_url, 'POST', '/api/match', json=payload)

    assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    return f"400 Bad Request (expected)"


def test_predict(client, base_url):
    """Test POST /api/predict"""
    payload = {
        "user_skills": ["react", "javascript", "css"],
        "job_skills": ["react", "javascript", "css", "typescript", "nextjs", "git"]
    }
    response = make_request(client, base_url, 'POST', '/api/predict', json=payload)
    data = response.json()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert 'model_score' in data
    assert 'coverage_score' in data
    assert 'final_score' in data
    assert 'match_category' in data
    assert data['n_matched'] == 3
    assert data['n_required'] == 6
    assert data['coverage_score'] == 0.5

    return f"200 OK | final={data['final_score']:.3f} | {data['match_category']}"


def test_normalize(client, base_url):
    """Test POST /api/normalize"""
    payload = {
        "skills": ["React.js", "Node.js", "C++", "PostgreSQL", "CI/CD", "UnknownXYZ"]
    }
    response = make_request(client, base_url, 'POST', '/api/normalize', json=payload)
    data = response.json()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert 'react' in data['normalized']
    assert 'node' in data['normalized']
    assert 'cpp' in data['normalized']
    assert 'sql' in data['normalized']
    assert 'cicd' in data['normalized']
    assert 'UnknownXYZ' in data['unknown']
    assert data['total_recognized'] == 5
    assert data['total_unknown'] == 1

    return f"200 OK | recognized={data['total_recognized']} | unknown={data['total_unknown']}"


def test_extract_cv(client, base_url):
    """Test POST /api/extract-cv"""
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testing_cv.pdf')

    if not os.path.exists(pdf_path):
        return "SKIP (testing_cv.pdf not found)"

    with open(pdf_path, 'rb') as f:
        if client:
            response = make_request(
                client, base_url, 'POST', '/api/extract-cv',
                files={"file": ("testing_cv.pdf", f, "application/pdf")}
            )
        else:
            response = requests.post(
                f"{base_url}/api/extract-cv",
                files={"file": ("testing_cv.pdf", f, "application/pdf")}
            )

    data = response.json()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data['success'] == True
    assert data['total_skills_found'] > 0
    assert len(data['extracted_skills']) > 0

    return f"200 OK | {data['total_skills_found']} skills | method={data['extraction_method']}"


def test_extract_cv_invalid_file(client, base_url):
    """Test POST /api/extract-cv dengan file bukan PDF"""
    # Buat fake file .txt
    fake_content = b"This is not a PDF file"

    if client:
        response = make_request(
            client, base_url, 'POST', '/api/extract-cv',
            files={"file": ("fake.txt", fake_content, "text/plain")}
        )
    else:
        response = requests.post(
            f"{base_url}/api/extract-cv",
            files={"file": ("fake.txt", fake_content, "text/plain")}
        )

    assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    return f"400 Bad Request (expected)"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MAGANG-IN AI — API TEST SUITE")
    print("=" * 70)
    print()

    # Determine base URL
    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    if base_url:
        print(f"Testing against: {base_url}")
        print("Using: requests library")
    else:
        print("Testing with: FastAPI TestClient (no server needed)")
    print()

    # Get client
    client, base_url = get_client(base_url)

    # Run tests
    tests = [
        ("GET  /api/health", test_health),
        ("GET  /api/skills", test_skills),
        ("POST /api/match", test_match),
        ("POST /api/match (unknown skills)", test_match_unknown_skills),
        ("POST /api/match (all invalid)", test_match_empty_skills),
        ("POST /api/predict", test_predict),
        ("POST /api/normalize", test_normalize),
        ("POST /api/extract-cv", test_extract_cv),
        ("POST /api/extract-cv (invalid)", test_extract_cv_invalid_file),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        try:
            result = test_fn(client, base_url)
            if result.startswith("SKIP"):
                print(f"  ⏭ {name:<40} {result}")
                skipped += 1
            else:
                print(f"  ✓ {name:<40} {result}")
                passed += 1
        except AssertionError as e:
            print(f"  ✗ {name:<40} FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name:<40} ERROR: {e}")
            failed += 1

    # Summary
    print()
    print("-" * 70)
    total = passed + failed + skipped
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped / {total} total")

    if failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"WARNING: {failed} test(s) failed!")
        sys.exit(1)

    print("=" * 70)
