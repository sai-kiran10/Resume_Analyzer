import streamlit as st
from io import BytesIO
import docx2txt
from PyPDF2 import PdfReader
import spacy
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Setup ---
st.set_page_config(page_title="Resume Analyzer", layout="wide")

# --- Load models ---
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load skill list ---
def load_skills():
    with open("skills/skills_list.txt", "r", encoding="utf-8") as f:
        return set(s.strip().lower() for s in f.read().splitlines() if s.strip())

SKILL_SET = load_skills()

# --- Synonyms for semantic expansion ---
SYNONYMS = {
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing"
}

def expand_skills(skills):
    expanded = set()
    for s in skills:
        expanded.add(s)
        if s in SYNONYMS:
            expanded.add(SYNONYMS[s])
    return expanded

# --- Education parsing ---
DEGREE_PATTERNS = {
    "bachelor": r"(bachelor|b\.tech|bsc|b\.s\.|undergraduate)",
    "master": r"(master|m\.tech|msc|m\.s\.)",
    "phd": r"(phd|doctorate)"
}

def extract_education(text):
    text_lower = text.lower()
    found_degrees = []
    for deg, pattern in DEGREE_PATTERNS.items():
        if re.search(pattern, text_lower):
            found_degrees.append(deg)
    return found_degrees

def match_education(resume_degrees, jd_text):
    jd_text = jd_text.lower()
    for deg in DEGREE_PATTERNS.keys():
        if deg in jd_text and deg in resume_degrees:
            return True
    return False

# --- File parsing ---
def extract_text_from_pdf(file_bytes: BytesIO) -> str:
    reader = PdfReader(file_bytes)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def extract_text_from_docx(file_bytes: BytesIO) -> str:
    return docx2txt.process(file_bytes) or ""

# --- Skill extraction ---
def extract_skills(text: str):
    text_lower = text.lower()
    found = {skill for skill in SKILL_SET if re.search(rf"\b{re.escape(skill)}\b", text_lower)}
    return sorted(found)

# --- Similarity ---
def compute_similarity(text1: str, text2: str):
    emb1 = embedder.encode([text1])
    emb2 = embedder.encode([text2])
    return round(float(cosine_similarity(emb1, emb2)[0][0]) * 100, 2)

# --- Weighted scoring ---
def compute_weighted_score(skill_fraction, edu_match, similarity_score):
    skill_score = skill_fraction * 70  # skills 70%
    edu_score = 10 if edu_match else 0 # education 10%
    sim_score = similarity_score * 0.2 # similarity 20%
    return round(skill_score + edu_score + sim_score, 2)

# --- Streamlit UI ---
st.title("Resume Analyzer")

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

with col2:
    jd_option = st.radio("Job Description Input:", ["Upload file", "Paste text"], horizontal=True)
    jd_text = ""
    if jd_option == "Upload file":
        jd_file = st.file_uploader("📝 Upload Job Description (TXT/DOCX/PDF)", type=["txt", "docx", "pdf"])
        if jd_file:
            b = BytesIO(jd_file.read())
            if jd_file.name.endswith(".pdf"):
                jd_text = extract_text_from_pdf(b)
            elif jd_file.name.endswith(".docx"):
                jd_text = extract_text_from_docx(b)
            else:
                jd_text = b.read().decode(errors="ignore")
    else:
        jd_text = st.text_area("Paste Job Description here:", height=200)

resume_text = ""
if resume_file:
    b = BytesIO(resume_file.read())
    if resume_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(b)
    else:
        resume_text = extract_text_from_docx(b)

if resume_text and jd_text.strip():
    st.success("✅ Ready to analyze match.")

    # --- Skills ---
    resume_skills = expand_skills(extract_skills(resume_text))
    jd_skills = expand_skills(extract_skills(jd_text))
    matched_skills = resume_skills & jd_skills
    missing_skills = jd_skills - resume_skills
    skill_fraction = len(matched_skills) / len(jd_skills) if jd_skills else 0

    # --- Education ---
    resume_degrees = extract_education(resume_text)
    edu_match = match_education(resume_degrees, jd_text)

    # --- Similarity ---
    similarity_score = compute_similarity(resume_text, jd_text)

    # --- Weighted score ---
    overall_score = compute_weighted_score(skill_fraction, edu_match, similarity_score / 100)

    # --- Display ---
    st.subheader("📊 Match Analysis")
    st.metric("Overall Weighted Score", f"{overall_score}%")
    st.write(f"**Skills Matched ({len(matched_skills)}):** {', '.join(sorted(matched_skills)) or 'None'}")
    st.write(f"**Missing Skills ({len(missing_skills)}):** {', '.join(sorted(missing_skills)) or 'None'}")
    st.write(f"**Education Match:** {'Yes' if edu_match else 'No'}")
    st.write(f"**Semantic Similarity:** {similarity_score}%")

    with st.expander("🔍 View Job Description Text"):
        st.text(jd_text[:5000])
    with st.expander("📘 View Resume Text"):
        st.text(resume_text[:5000])

else:
    st.info("Please upload a resume and either upload or paste a job description.")
