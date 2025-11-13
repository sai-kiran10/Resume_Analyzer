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

def show_signature():
    print("⚡ Resume Analyzer by Sai Kiran Vasa")

show_signature()

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
            "JD Input Type",
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
        jd_text = st.text_area("Paste Job Description Here:", height=200)

# --- Center the Parse & Analyze button.
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

# --- Analysis & Enhanced UI ---
if analyze and resume_text and jd_text and jd_text.strip():
    #st.success("✅ Analyzing resume and job description...")

    # --- Compute analysis (skills, education, similarity, score) ---
    resume_skills = expand_skills(extract_skills(resume_text))
    jd_skills = expand_skills(extract_skills(jd_text))
    matched_skills = resume_skills & jd_skills
    missing_skills = jd_skills - resume_skills
    skill_fraction = len(matched_skills) / len(jd_skills) if jd_skills else 0.0

    resume_degrees = extract_education(resume_text)
    edu_match = match_education(resume_degrees, jd_text)

    similarity_score = compute_similarity(resume_text, jd_text)  # percentage 0-100
    overall_score = compute_weighted_score(skill_fraction, edu_match, similarity_score / 100)

    # --- Top: Big overall score and status label ---
    if overall_score >= 90:
        overall_color = "#16a34a"  # green
        status_text = "Strong Match 💪"
    elif overall_score >= 80:
        overall_color = "#059669"  # light-green
        status_text = "Good Match 👍"
    elif overall_score >= 70:
        overall_color = "#f59e0b"  # amber
        status_text = "Fair Match"
    else:
        overall_color = "#dc2626"  # red
        status_text = "Needs Improvement 🚀"

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:6px;">
            <div style="font-size:64px; font-weight:800; color:{overall_color};">{overall_score}%</div>
            <div style="font-size:20px; color:gray; margin-top:4px;">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Breakdown Analysis")

    # --- Three interactive "cards" implemented as expanders (persistent) ---
    c1, c2, c3 = st.columns(3)

    # Education card
    with c1:
        header = "🎓 Education"
        sub = "Matched" if edu_match else "Not matched"
        st.markdown(f"**{header} — {sub}**")
        with st.expander("Details"):
            if edu_match:
                st.success("Candidate satisfies the degree requirement stated in the JD.")
                if resume_degrees:
                    st.write("Degrees found in resume:", ", ".join(resume_degrees))
            else:
                st.error("Candidate does NOT satisfy the degree requirement (per JD).")
                if resume_degrees:
                    st.write("Degrees found in resume:", ", ".join(resume_degrees))
                else:
                    st.write("No degree mentions detected in resume.")

    # Skills card
    with c2:
        header = "🧠 Skills"
        st.markdown(f"**{header} — {len(matched_skills)}/{len(jd_skills) if jd_skills else 0} matched**")
        with st.expander("View matched / missing skills"):
            if matched_skills:
                # matched badges
                matched_html = " ".join([f"<span style='background:#16a34a;color:white;padding:6px 8px;border-radius:6px;margin:3px;display:inline-block;font-size:13px;'>{m}</span>" for m in sorted(matched_skills)])
                st.markdown(f"**Matched:**<br>{matched_html}", unsafe_allow_html=True)
            else:
                st.write("No matched skills found.")

            if missing_skills:
                missing_html = " ".join([f"<span style='background:#ef4444;color:white;padding:6px 8px;border-radius:6px;margin:3px;display:inline-block;font-size:13px;'>{m}</span>" for m in sorted(missing_skills)])
                st.markdown(f"**Missing:**<br>{missing_html}", unsafe_allow_html=True)
            else:
                st.write("No missing skills detected.")

    # Semantic similarity card
    with c3:
        header = "🤖 Semantic Match"
        st.markdown(f"**{header} — {similarity_score}%**")
        with st.expander("Why this matters / Examples"):
            st.write("This score measures overall text similarity (context, experience descriptions).")
            # Optionally show short example matches: find top matching sentences (simple heuristic)
            try:
                # extract candidate sentences that are most similar to JD by embedding each sentence
                jd_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', jd_text) if s.strip()]
                resume_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', resume_text) if s.strip()]
                if jd_sentences and resume_sentences:
                    jd_embs = embedder.encode(jd_sentences)
                    res_embs = embedder.encode(resume_sentences)
                    import numpy as np
                    sims = cosine_similarity(jd_embs, res_embs)  # jd_sent x res_sent
                    # find top jd sentence -> best resume sentence pair
                    top_pairs = []
                    for i in range(min(3, len(jd_sentences))):
                        idx = sims[i].argmax()
                        top_pairs.append((jd_sentences[i][:200], resume_sentences[idx][:200], float(sims[i][idx])))
                    st.write("Sample matched snippets (JD → Resume):")
                    for jd_snip, res_snip, sc in top_pairs:
                        st.markdown(f"- **JD:** {jd_snip}")
                        st.markdown(f"  - **Resume:** {res_snip} (sim={sc:.2f})")
                else:
                    st.write("Not enough text to display snippet matches.")
            except Exception:
                st.write("Preview of snippet matches unavailable (embedding step skipped).")

    st.markdown("<br>", unsafe_allow_html=True)

# old condition to warn if analyze pressed but inputs missing
elif analyze:
    st.warning("⚠️ Please upload a resume and either upload or paste a job description before analyzing.")

st.markdown(
    "<hr><p style='text-align:center; color: gray; font-size:12px;'>© 2025 Sai Kiran. All rights reserved.</p>",
    unsafe_allow_html=True
)
