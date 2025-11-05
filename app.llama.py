import streamlit as st, os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()
persist_dir = os.getenv("PERSIST_DIR", "embeddings")

st.set_page_config(page_title="📚 Study Assistant", layout="wide")
st.title("📖 Departmental Exam RAG (LlamaIndex + OpenRouter)")

if "history" not in st.session_state:
    st.session_state.history = []

if not os.path.exists(persist_dir):
    st.warning("No index found. Run rag_llama.py first.")
else:
    embed_model = OpenAIEmbedding(
        model=os.getenv("OPENROUTER_EMBED_MODEL"),
        api_base=os.getenv("OPENROUTER_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    llm = OpenAI(
        model=os.getenv("OPENROUTER_MODEL"),
        api_base=os.getenv("OPENROUTER_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    chat_engine = index.as_chat_engine(llm=llm, chat_mode="context")

    query = st.chat_input("Ask a question from your PDFs...")
    if query:
        with st.spinner("Thinking..."):
            ans = chat_engine.chat(query)
        st.session_state.history.append({"q": query, "a": ans.response})
    for h in reversed(st.session_state.history):
        st.markdown(f"**Q:** {h['q']}")
        st.markdown(f"**A:** {h['a']}")
        st.markdown("---")
