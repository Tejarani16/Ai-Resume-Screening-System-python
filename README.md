# Ai-Resume-Screening-System-python
AI-powered resume–job matching system built using Python and FastAPI. The application analyzes resumes and job descriptions using NLP and TF-IDF vectorization to compute a similarity match score, helping recruiters shortlist candidates efficiently.
# AI Resume Screening System
# Tech Stack: Python, FastAPI, NLP (spaCy), Scikit-learn

from fastapi import FastAPI, UploadFile, File
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(title="AI Resume Screening System")

nlp = spacy.load("en_core_web_sm")

# Utility functions

def extract_text(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore")

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

@app.post("/match")
async def match_resume(resume: UploadFile = File(...), job_description: UploadFile = File(...)):
    resume_text = preprocess(extract_text(await resume.read()))
    jd_text = preprocess(extract_text(await job_description.read()))

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    return {
        "resume": resume.filename,
        "job_description": job_description.filename,
        "match_score": round(float(similarity_score) * 100, 2)
    }

# Run using: uvicorn app:app --reload
