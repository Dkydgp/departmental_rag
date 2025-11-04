# app.py
import streamlit as st
from utils import get_retriever, make_qa_chain
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="📚 Departmental Exam RAG Assistant", layout="wide")
st.title("📖 Departmental Exam Study Assistant (RAG + OpenRouter)")

if "history" not in st.session_state:
    st.session_state.history = []

k = st.sidebar.slider("Number of context chunks (k)", 2, 8, 4)
temperature = st.sidebar.slider("Response creativity", 0.0, 1.0, 0.2, 0.05)

retriever = get_retriever(k=k)
qa = make_qa_chain(retriever, temperature=temperature)

query = st.chat_input("Ask a question from your books...")

if query:
    with st.spinner("🔍 Reading your books..."):
        res = qa({"query": query})
    answer = res["result"]
    docs = res.get("source_documents", [])
    st.session_state.history.append({"query": query, "answer": answer, "sources": docs})

for h in reversed(st.session_state.history):
    st.markdown(f"**Q:** {h['query']}")
    st.markdown(f"**A:** {h['answer']}")
    if h["sources"]:
        st.markdown("**📚 Sources:**")
        for d in h["sources"][:3]:
            src = d.metadata.get("source_file", "unknown")
            page = d.metadata.get("page", d.metadata.get("page_number", ""))
            st.markdown(f"- {src} (page {page}) — `{d.page_content[:180]}...`")
    st.markdown("---")
