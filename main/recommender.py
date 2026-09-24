"""
main/recommender.py

Content-based recommendation engine using TF-IDF + cosine similarity,
replacing the LLM-prompt-based "recommendations" that were previously
sent to Groq.

Why TF-IDF instead of another transformer model: this task doesn't need
deep semantic understanding, just "which text documents are most similar
to this other text document" - which is exactly what a vector space
model with TF-IDF weighting is designed for. It's fast, runs on CPU with
no GPU needed, requires no training/fine-tuning, and is easy to explain
and justify in an FYP report as a standard, well-understood IR technique.

Two use cases:
1. recommend_courses()  - student profile vs. real course catalog (DB-grounded)
2. recommend_jobs()     - student profile vs. real Adzuna job listings
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Course, Material


# ---------------------------------------------------------------------------
# Course recommendations
# ---------------------------------------------------------------------------

def _build_course_profile(course):
    """Concatenate a course's name, department, and material descriptions into one text blob."""
    parts = [course.name]

    if course.department:
        parts.append(course.department.name)
        if course.department.description:
            parts.append(course.department.description)

    materials = Material.objects.filter(course_code=course)
    for material in materials:
        if material.description:
            parts.append(material.description)

    return " ".join(parts)


def recommend_courses(student, student_info_text="", top_n=3):
    """
    Recommend courses for a student using content-based filtering.

    student: a Student model instance (used to exclude already-enrolled courses
             and, if student_info_text is empty, to build a profile from their
             enrolled courses' content instead).
    student_info_text: optional free-text description of interests. If provided,
             it's combined with the student's enrolled-course content.
    top_n: how many courses to recommend.

    Returns a list of dicts: [{"course": Course, "score": float}, ...]
    """
    enrolled_courses = list(student.course.all()) if student else []
    enrolled_codes = {c.code for c in enrolled_courses}

    all_courses = list(Course.objects.all())
    candidate_courses = [c for c in all_courses if c.code not in enrolled_codes]

    if not candidate_courses:
        return []

    # Build the student's profile: their typed interests + content from courses
    # they're already enrolled in (their "history"), so recommendations reflect
    # both stated interest and demonstrated engagement.
    profile_parts = [student_info_text] if student_info_text else []
    for c in enrolled_courses:
        profile_parts.append(_build_course_profile(c))
    student_profile_text = " ".join(profile_parts).strip()

    if not student_profile_text:
        # No info at all to go on - can't meaningfully rank, so just
        # return nothing rather than a misleading similarity score.
        return []

    candidate_profiles = [_build_course_profile(c) for c in candidate_courses]

    # The student's profile is included as an extra "document" so it gets
    # vectorized in the same space as the courses, using the same vocabulary.
    corpus = candidate_profiles + [student_profile_text]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    student_vector = tfidf_matrix[-1]
    course_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(student_vector, course_vectors)[0]

    ranked = sorted(
        zip(candidate_courses, similarities), key=lambda pair: pair[1], reverse=True
    )

    return [{"course": course, "score": round(float(score), 4)} for course, score in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Job recommendations
# ---------------------------------------------------------------------------

def _job_text(job):
    return " ".join([
        job.get("title", ""),
        job.get("description", ""),
        job.get("category", ""),
    ])


def _top_shared_terms(student_profile_text, job_text, vectorizer, num_terms=5):
    """
    Extract the top TF-IDF terms that appear in both the student profile and
    the job text - a lightweight, explainable stand-in for the LLM-written
    'why this is a good fit' explanation.
    """
    student_terms = set(vectorizer.build_analyzer()(student_profile_text))
    job_terms = set(vectorizer.build_analyzer()(job_text))
    shared = student_terms & job_terms

    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_
    shared_with_weight = [
        (term, idf[vocab[term]]) for term in shared if term in vocab
    ]
    shared_with_weight.sort(key=lambda pair: pair[1], reverse=True)

    return [term for term, _ in shared_with_weight[:num_terms]]


def recommend_jobs(student_profile_text, jobs_data, top_n=5):
    """
    Rank real job listings (already fetched from Adzuna) against a student's
    profile using TF-IDF + cosine similarity.

    jobs_data: list of job dicts as returned by get_real_time_jobs().
    Returns a list of dicts: each original job dict plus "match_score" and
    "matching_keywords".
    """
    if not jobs_data or not student_profile_text.strip():
        return []

    job_texts = [_job_text(job) for job in jobs_data]
    corpus = job_texts + [student_profile_text]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    student_vector = tfidf_matrix[-1]
    job_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(student_vector, job_vectors)[0]

    ranked = sorted(
        zip(jobs_data, job_texts, similarities), key=lambda triple: triple[2], reverse=True
    )

    results = []
    for job, job_text, score in ranked[:top_n]:
        keywords = _top_shared_terms(student_profile_text, job_text, vectorizer)
        result = dict(job)
        result["match_score"] = round(float(score), 4)
        result["matching_keywords"] = keywords
        results.append(result)

    return results
