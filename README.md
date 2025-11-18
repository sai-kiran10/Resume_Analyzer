📄✨ AI-Powered Resume Analyzer — Complete Documentation

Welcome to the AI-Powered Resume Analyzer — an intelligent system that evaluates how well a resume matches a job description using NLP, semantic similarity, skill mapping, and weighted scoring.

This project delivers a professional, ATS-style analysis with a beautiful Apple-style match ring, clean UI, and detailed breakdown insights.

---

🚀 Project Overview

The Resume Analyzer helps candidates and recruiters quickly determine resume–job alignment by analyzing:

 ✔ Skills match <br>
 ✔ Education match <br>
 ✔ Semantic/contextual match <br>
 ✔ Weighted overall score <br>
 ✔ Matched and missing skills

The system uses state-of-the-art embeddings, custom rules, and a clean, interactive interface built on Streamlit.

---

⭐ Key Features

1. 🔍 Smart Resume & JD Parsing
   - Upload Resume: PDF, DOCX
   - Job Description Input: Upload file or Paste text
   - Automatic extraction using:
     - 📄 PyPDF2
     - 📄 docx2txt

2. 🤖 AI-Driven Analysis
   - Skills extraction + synonym expansion
   - Education pattern detection
   - Semantic similarity using Sentence-BERT (MiniLM-L6-v2)
   - Weighted score:
     - 🧠 70% Skills
     - 🎓 10% Education
     - 📝 20% Semantic similarity

3. 🟢 Apple-Style Match Ring (UI Highlight)
   - A stunning circular score indicator that changes color:
     - 🟢 Strong Match
     - 🟡 Fair Match
     - 🔴 Needs Improvement

4. 📊 Clean Breakdown Analysis
   - 🎓 Education match
   - 🧠 Skills matched + missing
   - 🤖 Semantic similarity %

---

🛠️ Tech Stack

Category | Technology
---------|------------
Frontend | Streamlit
NLP | spaCy (en_core_web_sm)
Embeddings | Sentence Transformers (MiniLM-L6-v2)
Similarity | cosine_similarity (sklearn)
Text Extraction | PyPDF2, docx2txt
Styling | Custom CSS
Backend | Python 3

---

⚙️ Installation Guide

1️⃣ Clone the Project <br>
git clone https://github.com/sai-kiran10/Resume_Analyzer.git <br>
cd resume-analyzer

2️⃣ Create Virtual Environment <br>
python -m venv venv <br>
venv/Scripts/activate    #Windows <br>
source venv/bin/activate  #Linux/macOS

3️⃣ Install Requirements <br>
pip install -r requirements.txt 

4️⃣ Install SpaCy Model <br>
python -m spacy download en_core_web_sm

---

▶️ Run the Application

Start the Streamlit server: <br>
streamlit run app.py

App opens at: <br>
http://localhost:8501

---

🧩 How It Works – Step by Step

Step 1 — Upload Resume
- PDF or DOCX format

Step 2 — Add Job Description
- Upload JD file or Paste JD manually

Step 3 — Click “Parse & Analyze”
- Computes matched skills, missing skills, detected degrees, semantic similarity (0–100%), weighted overall match

Step 4 — View Results
- 🎯 Beautiful Apple-styled score ring
- 🎓 Education match verdict
- 🧠 Skills matched/missing
- 🤖 Semantic similarity with examples

---

📊 Weighted Scoring Formula

Factor | Weight | Example
-------|--------|--------
Skills | 70% | 12/15 → 80%
Education | 10% | Match → +10
Semantic | 20% | 82% → 16.4

Final Score:
overall = (skills * 70) + (education ? 10 : 0) + (semantic * 0.2)

---

🔮 Future Scope

Planned Enhancements:
- 🧭 Skill Graph Visualization
- 🤝 Multi-Resume Comparison
- 🧾 Resume Rewrite Suggestions (AI-Generated)
- 📊 Interactive ATS Score Simulation
- 🔎 JD Auto-Category Detection (ML)
- 📝 Auto-Recommend Skills to Add

---

👤 Author

👨‍💻 Sai Kiran Vasa   
© 2025 — All Rights Reserved
