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
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

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
st.markdown("<h2 style='text-align:center;'>📄 Resume Analyzer</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Layout: two columns for uploads
# We'll use a spacer div before each uploader so the actual "Browse files" area lines up.
col_left, col_right = st.columns(2)

# Spacer height in px — tweak if you want the uploader boxes lower/higher
SPACER_PX1 = 70
SPACER_PX2 = 00
with col_left:
    st.markdown("<h4 style='text-align:center; margin-bottom:6px;'>Upload Resume</h4>", unsafe_allow_html=True)
    # spacer to push the uploader down so the browse-box aligns with the other column
    st.markdown(f"<div style='height:{SPACER_PX1}px'></div>", unsafe_allow_html=True)
    resume_file = st.file_uploader(" ", type=["pdf", "docx"], label_visibility="collapsed")

with col_right:
    st.markdown("<h4 style='text-align:center; margin-bottom:6px;'>Upload or Paste Job Description</h4>", unsafe_allow_html=True)
    #jd_option = st.radio("Input Type:", ["Upload Job Description File", "Paste Job Description"], horizontal=True, label_visibility="collapsed")
    
    # Center the radio buttons within the column
    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        jd_option = st.radio(
            "",
            ["Upload Job Description File", "Paste Job Description"],
            horizontal=True,
            label_visibility="collapsed"
        )

    # same spacer to align the JD uploader box with resume uploader
    st.markdown(f"<div style='height:{SPACER_PX2}px'></div>", unsafe_allow_html=True)
    
    jd_text = ""
    jd_file = None
    if jd_option == "Upload Job Description File":
        jd_file = st.file_uploader(" ", type=["txt", "docx", "pdf"], label_visibility="collapsed")
        if jd_file:
            b = BytesIO(jd_file.read())
            if jd_file.name.endswith(".pdf"):
                jd_text = extract_text_from_pdf(b)
            elif jd_file.name.endswith(".docx"):
                jd_text = extract_text_from_docx(b)
            else:
                jd_text = b.read().decode(errors="ignore")
    else:
        # when pasting, we want the text area to appear *below* the uploader baseline so the layout stays aligned
        jd_text = st.text_area("Paste Job Description Here:", height=200)

# --- Center the Parse & Analyze button.
# We'll create three equal columns and place the button in the center one.
btn_col_left, btn_col_center, btn_col_right = st.columns([1, 1, 1])

# style button via CSS for consistent appearance
st.markdown(
    """
    <style>
    /* Style the Streamlit button (applies to the next button rendered) */
    div.stButton > button {
        background-color: #2196F3;
        color: white;
        padding: 10px 30px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #0b7dda;
    }
    /* Reduce top/bottom padding to make layout tighter */
    section[data-testid="stSidebar"] { padding-top: 0rem; }
    </style>
    """, unsafe_allow_html=True
)

with btn_col_center:
    analyze = st.button("🚀 Parse & Analyze")

# --- Analysis Logic ---
resume_text = ""
if resume_file:
    b = BytesIO(resume_file.read())
    if resume_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(b)
    else:
        resume_text = extract_text_from_docx(b)

# If user uploaded JD file earlier, jd_text is already set; if they pasted, jd_text from text_area
if analyze and resume_text and jd_text and jd_text.strip():
    st.success("✅ Analyzing resume and job description...")

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

elif analyze:
    st.warning("⚠️ Please upload a resume and either upload or paste a job description before analyzing.")
