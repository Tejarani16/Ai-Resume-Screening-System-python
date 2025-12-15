# Ai-Resume-Screening-System-python
AI-powered resume–job matching system built using Python and FastAPI. The application analyzes resumes and job descriptions using NLP and TF-IDF vectorization to compute a similarity match score, helping recruiters shortlist candidates efficiently.
AI Resume Screening System

An AI-powered resume–job matching system built using Python and FastAPI.
The application analyzes resumes and job descriptions using NLP and TF-IDF vectorization to compute a similarity match score, helping recruiters shortlist candidates efficiently.

🚀 Features

Resume & Job Description upload (text-based)

NLP preprocessing using spaCy

Skill/context similarity using TF-IDF + Cosine Similarity

REST API built with FastAPI

JSON-based response with match percentage

Lightweight, fast, and scalable backend

🛠 Tech Stack

Python 3.9+

FastAPI

spaCy (NLP)

Scikit-learn

Uvicorn

📂 Project Structure
ai-resume-screening/
│── app.py
│── requirements.txt
│── README.md

⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-username/ai-resume-screening.git
cd ai-resume-screening

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\\Scripts\\activate     # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

4️⃣ Run Application
uvicorn app:app --reload

🔍 API Usage
Endpoint
POST /match

Input

resume → Resume text file (.txt)

job_description → Job description text file (.txt)

Response
{
  "resume": "resume.txt",
  "job_description": "jd.txt",
  "match_score": 78.45
}

🧠 How It Works

Text extraction from uploaded files

NLP preprocessing (lemmatization, stop-word removal)

TF-IDF vectorization

Cosine similarity computation

Match score generation (0–100%)

📌 Use Cases

HR resume shortlisting

ATS system enhancement

Talent analytics platforms

Recruitment automation

🔮 Future Enhancements

PDF/DOCX resume parsing

Skill extraction with NER

LLM-based semantic matching

Database integration

UI dashboard

👨‍💻 Author

Teja Rani
Python Developer | AI/ML | Backend Systems









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
