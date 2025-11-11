import streamlit as st
from io import BytesIO
import docx2txt
from PyPDF2 import PdfReader

st.set_page_config(page_title="Resume Analyzer - Phase 1", layout="wide")

st.title("Resume Analyzer — Phase 1: Upload & Parse")

uploaded = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx", "txt"])

def extract_text_from_pdf(file_bytes: BytesIO) -> str:
    try:
        reader = PdfReader(file_bytes)
        text = []
        for p in reader.pages:
            text.append(p.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        return f"[PDF parse error] {e}"

def extract_text_from_docx(file_bytes: BytesIO) -> str:
    try:
        # docx2txt expects a path or file-like object; BytesIO works
        text = docx2txt.process(file_bytes)
        return text or ""
    except Exception as e:
        return f"[DOCX parse error] {e}"

if uploaded:
    file_type = uploaded.type
    st.subheader(f"File: {uploaded.name} — {file_type}")

    # Read into BytesIO (streamlit gives UploadedFile which is file-like)
    bytes_io = BytesIO(uploaded.read())

    raw_text = ""
    if uploaded.name.lower().endswith(".pdf") or file_type == "application/pdf":
        raw_text = extract_text_from_pdf(bytes_io)
    elif uploaded.name.lower().endswith(".docx") or file_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",):
        # docx2txt expects a filename or file-like; ensure pointer at start
        bytes_io.seek(0)
        raw_text = extract_text_from_docx(bytes_io)
    elif uploaded.name.lower().endswith(".txt") or file_type == "text/plain":
        bytes_io.seek(0)
        raw_text = bytes_io.read().decode(errors="ignore")
    else:
        raw_text = "[Unsupported file type]"

    if not raw_text.strip():
        st.warning("No text extracted — resume may contain scanned images. Consider adding OCR later (Tesseract).")
    else:
        st.success("Text extracted successfully.")
        with st.expander("Show extracted text (first 10k chars)"):
            st.text(raw_text[:10000])

        # quick preview: lines, word count
        lines = [ln for ln in raw_text.splitlines() if ln.strip()]
        st.metric("Approx. words", len(raw_text.split()))
        st.metric("Approx. non-empty lines", len(lines))

        # Save parsed to local cache if you want (simple)
        if st.button("Save parsed text to local file"):
            out_name = uploaded.name.rsplit(".", 1)[0] + "_parsed.txt"
            with open(out_name, "w", encoding="utf-8") as f:
                f.write(raw_text)
            st.info(f"Saved parsed text as `{out_name}` in working directory.")
