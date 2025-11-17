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
    # console signature (visible in terminal)
    print("⚡ Resume Analyzer by Sai Kiran Vasa")

show_signature()

# ---------------------
# --- CSS / Styling ---
# ---------------------
st.markdown("""
<style>
/* Apple-style activity ring (conic + subtle gradient + shadow) */
.activity-ring {
  width: 220px;
  height: 220px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: auto;
  position: relative;
  box-shadow: 0 6px 18px rgba(15,23,42,0.08);
}

/* layered rings using pseudo technique via inner wrappers */
.activity-ring .outer {
  width: 220px;
  height: 220px;
  border-radius: 50%;
  background: conic-gradient(var(--color1) 0deg, var(--color1) calc(var(--angle)), rgba(0,0,0,0) calc(var(--angle)));
  display:flex;
  align-items:center;
  justify-content:center;
  filter: drop-shadow(0 1px 2px rgba(0,0,0,0.08));
}

/* inner gradient overlay to make it look like Apple ring */
.activity-ring .outer::after {
  content: "";
  position: absolute;
  width: 220px;
  height: 220px;
  border-radius: 50%;
  background: conic-gradient(rgba(255,255,255,0.12) 0deg, rgba(255,255,255,0) 120deg);
  opacity: 0.6;
  border-radius:50%;
}

/* inner white circle that contains the number */
.activity-ring .inner {
  width: 150px;
  height: 150px;
  background: linear-gradient(180deg, #ffffff, #fbfbfb);
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  box-shadow: inset 0 -6px 12px rgba(0,0,0,0.03);
}

/* score text */
.activity-ring .score {
  font-size: 44px;
  font-weight: 800;
  margin-bottom: 6px;
}

.activity-ring .status {
  font-size: 16px;
  color: #6b7280;
}

/* small metric cards (styled expanders) */
.metrics-row {
  margin-top: 18px;
  margin-bottom: 6px;
}

.metric-card {
  background: #f8fafc;
  border-radius: 12px;
  padding: 14px;
  text-align: center;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.12s ease-in-out;
  border: 1px solid #eceff1;
}
.metric-card:hover {
  background: #eef2f6;
}

/* badges */
.badge-match {
  background:#10b981;
  color:white;
  padding:6px 8px;
  border-radius:6px;
  margin:3px;
  display:inline-block;
  font-size:13px;
}
.badge-miss {
  background:#ef4444;
  color:white;
  padding:6px 8px;
  border-radius:6px;
  margin:3px;
  display:inline-block;
  font-size:13px;
}

/* slight responsive tweak: keep columns tidy */
@media (max-width: 800px) {
  .activity-ring { width: 180px; height: 180px; }
  .activity-ring .outer { width: 180px; height: 180px; }
  .activity-ring .inner { width: 120px; height: 120px; }
  .activity-ring .score { font-size: 36px; }
}
</style>
""", unsafe_allow_html=True)

# --- Streamlit UI ---
st.markdown("<h2 style='text-align:center;'>📄 Resume Analyzer</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Layout: two columns for uploads
col_left, col_right = st.columns(2)

# Spacer height in px — tweak if you want the uploader boxes lower/higher
SPACER_PX1 = 70
SPACER_PX2 = 0
with col_left:
    st.markdown("<h4 style='text-align:center; margin-bottom:6px;'>Upload Resume</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='height:{SPACER_PX1}px'></div>", unsafe_allow_html=True)
    resume_file = st.file_uploader(" ", type=["pdf", "docx"], label_visibility="collapsed")

with col_right:
    st.markdown("<h4 style='text-align:center; margin-bottom:6px;'>Upload or Paste Job Description</h4>", unsafe_allow_html=True)
    # center radio inside column
    col_l, col_mid, col_r = st.columns([1, 3, 1])
    with col_mid:
        jd_option = st.radio(
            "JD Input Type",
            ["Upload Job Description File", "Paste Job Description"],
            horizontal=True,
            label_visibility="collapsed"
        )
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
st.markdown(
    """
    <style>
    div.stButton > button.primary-button {
        background-color: #2563eb;
        color: white;
        padding: 10px 34px;
        border-radius: 10px;
        border: none;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
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
    # compute analysis
    with st.spinner("Analyzing... (embeddings may take a few seconds)"):
        resume_skills = expand_skills(extract_skills(resume_text))
        jd_skills = expand_skills(extract_skills(jd_text))
        matched_skills = resume_skills & jd_skills
        missing_skills = jd_skills - resume_skills
        skill_fraction = len(matched_skills) / len(jd_skills) if jd_skills else 0.0

        resume_degrees = extract_education(resume_text)
        edu_match = match_education(resume_degrees, jd_text)

        similarity_score = compute_similarity(resume_text, jd_text)  # percentage 0-100
        overall_score = compute_weighted_score(skill_fraction, edu_match, similarity_score / 100)

    # decide ring color and status text
    if overall_score >= 90:
        ring_color = "#16a34a"
        status_text = "Strong Match 💪"
    elif overall_score >= 80:
        ring_color = "#059669"
        status_text = "Good Match 👍"
    elif overall_score >= 70:
        ring_color = "#f59e0b"
        status_text = "Fair Match"
    else:
        ring_color = "#ef4444"
        status_text = "Needs Improvement 🚀"

    # Render Apple-style activity ring around an inner white circle
    # -- we convert score to degrees (0-360)
    angle_deg = float(overall_score) * 3.6
    st.markdown(
        f"""
        <div class="activity-ring" style="--color1: {ring_color}; --angle: {angle_deg}deg;">
            <div class="outer" style="background: conic-gradient({ring_color} 0deg, {ring_color} calc({angle_deg}), rgba(0,0,0,0) calc({angle_deg}));">
                <div class="inner">
                    <div class="score" style="color:{ring_color};">{overall_score}%</div>
                    <div class="status">{status_text}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Breakdown Analysis")

    # Three columns for metrics (use expanders as interactive "cards")
    c1, c2, c3 = st.columns(3)

    # EDUCATION
    with c1:
        header_text = f"🎓 Education — {'Matched' if edu_match else 'Not matched'}"
        # style header as markdown to look like a card
        st.markdown(f"<div style='background:#f8fafc;padding:10px;border-radius:8px;text-align:center;font-weight:700'>{header_text}</div>", unsafe_allow_html=True)
        with st.expander("View education details"):
            st.write("**Detected Degrees:**", ", ".join(resume_degrees) if resume_degrees else "None detected")
            if edu_match:
                st.success("✔ Candidate satisfies the degree requirement.")
            else:
                st.error("✘ Candidate does NOT meet the required degree.")

    # SKILLS
    with c2:
        header_text = f"🧠 Skills — {len(matched_skills)}/{len(jd_skills) if jd_skills else 0} matched"
        st.markdown(f"<div style='background:#f8fafc;padding:10px;border-radius:8px;text-align:center;font-weight:700'>{header_text}</div>", unsafe_allow_html=True)
        with st.expander("View matched & missing skills"):
            st.markdown("**Matched Skills:**")
            if matched_skills:
                for m in sorted(matched_skills):
                    st.markdown(f"<span class='badge-match'>{m}</span>", unsafe_allow_html=True)
            else:
                st.write("None")

            st.markdown("**Missing Skills:**")
            if missing_skills:
                for m in sorted(missing_skills):
                    st.markdown(f"<span class='badge-miss'>{m}</span>", unsafe_allow_html=True)
            else:
                st.write("None")

    # SEMANTIC
    with c3:
        header_text = f"🤖 Semantic Match — {similarity_score}%"
        st.markdown(f"<div style='background:#f8fafc;padding:10px;border-radius:8px;text-align:center;font-weight:700'>{header_text}</div>", unsafe_allow_html=True)
        with st.expander("Why this matters / examples"):
            st.write("This score measures contextual similarity between the resume and the job description.")
            # show up to 3 JD -> resume snippet matches
            try:
                jd_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', jd_text) if s.strip()]
                resume_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', resume_text) if s.strip()]
                if jd_sentences and resume_sentences:
                    jd_embs = embedder.encode(jd_sentences)
                    res_embs = embedder.encode(resume_sentences)
                    import numpy as np
                    sims = cosine_similarity(jd_embs, res_embs)
                    top_pairs = []
                    for i in range(min(3, len(jd_sentences))):
                        idx = sims[i].argmax()
                        top_pairs.append((jd_sentences[i][:200], resume_sentences[idx][:200], float(sims[i][idx])))
                    st.write("Sample matched snippets (JD → Resume):")
                    for jd_snip, res_snip, sc in top_pairs:
                        st.markdown(f"- **JD:** {jd_snip}")
                        st.markdown(f"  - **Resume:** {res_snip} (sim={sc:.2f})")
                else:
                    st.write("Not enough text for snippet preview.")
            except Exception:
                st.write("Snippet preview unavailable.")

    st.markdown("<br>", unsafe_allow_html=True)

# warn if analyze pressed but missing inputs
elif analyze:
    st.warning("⚠️ Please upload a resume and either upload or paste a job description before analyzing.")

# Footer
st.markdown(
    "<hr><p style='text-align:center; color: gray; font-size:12px;'>© 2025 Sai Kiran. All rights reserved.</p>",
    unsafe_allow_html=True
)
