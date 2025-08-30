# app.py
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
from dotenv import load_dotenv
from io import BytesIO
import re

# Load env (OPENAI_API_KEY in .env)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Set OPENAI_API_KEY in a .env file or your environment variables.")
openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="StudyMate — PDF Q&A", layout="wide")

st.title("📚 StudyMate — PDF-based Q&A (Streamlit)")

# Initialize sentence-transformers model
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
model = load_embed_model()

# Helpers
def extract_text_from_pdf(uploaded_file) -> str:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    reader = PdfReader(BytesIO(raw))
    text_parts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)

def chunk_text(text, max_chars=1000, overlap=200):
    # split into sentences and accumulate into chunks up to max_chars (approx)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= max_chars:
            current += " " + sent
        else:
            if current:
                chunks.append(current.strip())
            if len(sent) > max_chars:
                # break long sentence
                for i in range(0, len(sent), max_chars - overlap):
                    chunks.append(sent[i:i + max_chars].strip())
                current = ""
            else:
                current = sent
    if current:
        chunks.append(current.strip())
    return chunks

def embed_texts(texts):
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # normalize embeddings for cosine similarity speed
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0] = 1e-9
    embeddings = embeddings / norms
    return embeddings

def retrieve(query, embeddings, chunks, top_k=4):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    sims = cosine_similarity(q_emb, embeddings)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    return idxs, sims[idxs]

def call_openai_chat(system_prompt, user_prompt, model_name="gpt-3.5-turbo", max_tokens=512, temperature=0.2):
    try:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role":"system", "content": system_prompt},
                {"role":"user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI request failed: {e}")
        return None

# Session state for index storage
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None
if "sources" not in st.session_state:
    st.session_state["sources"] = []  # names of PDFs or filenames

# Sidebar controls
with st.sidebar:
    st.header("Index Controls")
    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    max_chars = st.slider("Chunk size (chars)", min_value=300, max_value=2500, value=1000, step=100)
    overlap = st.slider("Chunk overlap (chars)", min_value=0, max_value=800, value=200, step=50)
    top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=8, value=4)
    model_choice = st.selectbox("OpenAI model", options=["gpt-3.5-turbo", "gpt-4"], index=0)
    temp = st.slider("Answer creativity (temperature)", 0.0, 1.0, 0.2, step=0.1)
    if st.button("Clear index"):
        st.session_state["chunks"] = []
        st.session_state["embeddings"] = None
        st.session_state["sources"] = []
        st.success("Index cleared.")

# Process uploaded PDFs
if uploaded:
    all_new_chunks = []
    source_names = []
    st.info(f"Processing {len(uploaded)} PDF(s)...")
    for file in uploaded:
        name = file.name
        source_names.append(name)
        text = extract_text_from_pdf(file)
        if not text.strip():
            st.warning(f"No selectable text detected in {name}. If it's scanned, OCR is needed.")
            continue
        new_chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
        st.write(f"➡️ {name}: extracted {len(new_chunks)} chunks.")
        # optionally add source info to chunk prefix
        new_chunks = [f"[{name} — chunk {i+1}]\n{c}" for i, c in enumerate(new_chunks)]
        all_new_chunks.extend(new_chunks)

    if all_new_chunks:
        with st.spinner("Computing embeddings..."):
            new_embeddings = embed_texts(all_new_chunks)
        # merge with existing
        if st.session_state["chunks"]:
            st.session_state["chunks"].extend(all_new_chunks)
            st.session_state["sources"].extend(source_names)
            st.session_state["embeddings"] = np.vstack([st.session_state["embeddings"], new_embeddings])
        else:
            st.session_state["chunks"] = all_new_chunks
            st.session_state["sources"] = source_names
            st.session_state["embeddings"] = new_embeddings
        st.success(f"Indexed {len(all_new_chunks)} chunks. Total chunks: {len(st.session_state['chunks'])}")

# Main query UI
st.subheader("Ask questions about your uploaded PDF(s)")
col1, col2 = st.columns([3,1])
with col1:
    query = st.text_area("Enter your question", height=120)
with col2:
    ask = st.button("Ask StudyMate")

if ask:
    if not query:
        st.warning("Type a question first.")
    elif not st.session_state["chunks"]:
        st.warning("No indexed PDFs. Upload a PDF first in the sidebar.")
    else:
        with st.spinner("Retrieving relevant passages..."):
            idxs, scores = retrieve(query, st.session_state["embeddings"], st.session_state["chunks"], top_k=top_k)
            retrieved = [st.session_state["chunks"][i] for i in idxs]
        st.markdown("**Retrieved passages (top results):**")
        for i, (r, s, idx) in enumerate(zip(retrieved, scores, idxs)):
            st.write(f"**Result {i+1}** (score: {s:.4f}, index: {int(idx)})")
            st.write(r[:600] + ("..." if len(r)>600 else ""))

        # Build prompt to OpenAI
        system_prompt = "You are StudyMate, an educational assistant. Use the provided context strictly to answer the question. Cite which retrieved passages you used (by index). If the answer is not in the context, say you don't know and suggest where to look."
        context_text = "\n\n---\n\n".join([f"Passage {int(idx)}:\n{st.session_state['chunks'][int(idx)]}" for idx in idxs])
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer concisely and mention which passages (by index) you used."
        with st.spinner("Generating answer from OpenAI..."):
            answer = call_openai_chat(system_prompt, user_prompt, model_name=model_choice, max_tokens=512, temperature=float(temp))

        if answer:
            st.markdown("### ✅ Answer")
            st.write(answer)
            # show sources used (best effort: parse numbers present in answer)
            st.markdown("### Sources (retrieved passages)")
            for i, idx in enumerate(idxs):
                st.write(f"- Passage index {int(idx)} — score {scores[i]:.4f}")
            # Download button
            combined_text = f"Question:\n{query}\n\nAnswer:\n{answer}\n\nRetrieved passages:\n{context_text}"
            st.download_button("Download Q&A", combined_text, file_name="studymate_answer.txt")

# Small status footer
st.sidebar.markdown("---")
st.sidebar.write(f"Indexed chunks: {len(st.session_state['chunks'])}")
st.sidebar.write("Tip: If your PDF is a scanned image, run OCR (Tesseract) before uploading.")

# Optional: save index to disk
if st.session_state["chunks"]:
    if st.sidebar.button("Save index to disk (npz)"):
        np.savez("studymate_index.npz", embeddings=st.session_state["embeddings"], chunks=np.array(st.session_state["chunks"], dtype=object))
        st.sidebar.success("Index saved to studymate_index.npz")