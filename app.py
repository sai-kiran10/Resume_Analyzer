import streamlit as st
from io import BytesIO
import docx2txt
from PyPDF2 import PdfReader
import spacy
import re

st.set_page_config(page_title="Resume Analyzer", layout="wide")

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Load skills list
def load_skills():
    with open("skills/skills_list.txt", "r", encoding="utf-8") as f:
        skills = [s.strip().lower() for s in f.read().splitlines() if s.strip()]
    return set(skills)

SKILL_SET = load_skills()

# --- File parsing functions ---
def extract_text_from_pdf(file_bytes: BytesIO) -> str:
    reader = PdfReader(file_bytes)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def extract_text_from_docx(file_bytes: BytesIO) -> str:
    text = docx2txt.process(file_bytes)
    return text or ""

# --- NLP + Skill Extraction ---
def extract_skills(text: str):
    text_lower = text.lower()
    found = {skill for skill in SKILL_SET if re.search(rf"\b{re.escape(skill)}\b", text_lower)}
    return sorted(found)

def extract_entities(text: str):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["ORG", "GPE", "DATE", "PERSON"]]
    return entities

# --- Streamlit UI ---
st.title("Resume Analyzer — Phase 2: Skill Extraction")

uploaded = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

if uploaded:
    bytes_io = BytesIO(uploaded.read())
    if uploaded.name.endswith(".pdf"):
        raw_text = extract_text_from_pdf(bytes_io)
    else:
        bytes_io.seek(0)
        raw_text = extract_text_from_docx(bytes_io)

    if not raw_text.strip():
        st.warning("No text extracted — might be an image-based resume.")
    else:
        st.success("Text extracted successfully.")
        with st.expander("Show Extracted Text"):
            st.text(raw_text[:5000])

        # Skill extraction
        skills = extract_skills(raw_text)
        entities = extract_entities(raw_text)

        st.subheader("🧠 Extracted Skills")
        st.write(", ".join(skills) if skills else "No skills detected.")

        st.subheader("🏢 Detected Entities (Organizations, Dates, etc.)")
        for ent, label in entities:
            st.write(f"**{ent}** — {label}")

        st.metric("Total Skills Found", len(skills))
