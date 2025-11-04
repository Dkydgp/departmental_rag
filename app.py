# app.py
import streamlit as st
from utils import get_retriever, make_qa_chain
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="RAG Study Assistant", layout="wide")
st.title("📚 RAG Study Assistant — Departmental Exam Prep")

if "history" not in st.session_state:
    st.session_state.history = []

k = st.sidebar.slider("Number of retrieved chunks (k)", min_value=1, max_value=8, value=4)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

retriever = get_retriever(k=k)
qa = make_qa_chain(retriever, temperature=temperature)

query = st.chat_input("Ask a question about your books...")

if query:
    with st.spinner("Searching your books..."):
        res = qa({"query": query})
    answer = res["result"]
    docs = res.get("source_documents", [])
    # display
    st.session_state.history.append({"query": query, "answer": answer, "sources": docs})
    st.success("Answer ready — see below ⬇️")

# show chat history
for h in reversed(st.session_state.history):
    st.markdown(f"**Q:** {h['query']}")
    st.markdown(f"**A:** {h['answer']}")
    if h["sources"]:
        st.markdown("**Sources:**")
        for d in h["sources"][:5]:
            src = d.metadata.get("source_file", "unknown")
            page = d.metadata.get("page", d.metadata.get("page_number", ""))
            # You may need to adapt keys depending on loader metadata structure
            st.markdown(f"- {src} (page: {page}) — _snippet_: {d.page_content[:200]}...")
    st.markdown("---")
