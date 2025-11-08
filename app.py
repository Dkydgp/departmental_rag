"""
app.py - Streamlit RAG Chat Interface
Interactive chat interface for querying indexed documents
"""

import os
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Headless mode (for Render)
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

load_dotenv()

# Configuration
PERSIST_DIR = os.getenv("PERSIST_DIR", "storage")

# Page config
st.set_page_config(
    page_title="📚 Study Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding-top: 2rem;}
    .stChatMessage {padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .user-message {background-color: #e3f2fd;}
    .assistant-message {background-color: #f5f5f5;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📖 Departmental Exam RAG Assistant")
st.markdown("*Ask questions about your study materials powered by AI*")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("📡 Model Configuration")
    embed_model = os.getenv('OPENROUTER_EMBED_MODEL', 'Not set')
    llm_model = os.getenv('OPENROUTER_MODEL', 'Not set')
    st.info(f"**Embedding:** {embed_model}\n\n**LLM:** {llm_model}")
    
    chat_mode = st.selectbox("Chat Mode", ["context", "condense_question", "simple"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    
    st.markdown("---")
    st.subheader("🔧 Actions")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.session_state.messages:
        export_text = "\n\n".join(
            [f"{'You' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in st.session_state.messages]
        )
        st.download_button(
            "💾 Download Chat",
            data=export_text,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Check for index
if not os.path.exists(PERSIST_DIR):
    st.error("❌ **No index found!**")
    st.info("Run `python rag_llama.py --build` and refresh the app.")
    st.stop()

# Load index once
if not st.session_state.index_loaded:
    try:
        with st.spinner("🔄 Loading index..."):
            required_vars = [
                "OPENROUTER_API_KEY",
                "OPENROUTER_MODEL",
                "OPENROUTER_EMBED_MODEL",
                "OPENROUTER_BASE_URL"
            ]
            missing = [v for v in required_vars if not os.getenv(v)]
            if missing:
                st.error(f"❌ Missing env variables: {', '.join(missing)}")
                st.stop()

            embed_model = OpenAIEmbedding(
                model=os.getenv("OPENROUTER_EMBED_MODEL"),
                api_base=os.getenv("OPENROUTER_BASE_URL"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context, embed_model=embed_model)

            llm = OpenAI(
                model=os.getenv("OPENROUTER_MODEL"),
                api_base=os.getenv("OPENROUTER_BASE_URL"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=temperature
            )

            st.session_state.chat_engine = index.as_chat_engine(
                llm=llm, chat_mode=chat_mode, verbose=False
            )
            st.session_state.index_loaded = True
            st.success("✅ System ready!")
    except Exception as e:
        st.error(f"❌ Error loading index: {e}")
        st.stop()

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                answer = response.response
                placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.markdown("---")
st.markdown(
    f"<p style='text-align:center;color:gray;font-size:0.9em;'>Powered by LlamaIndex + OpenRouter | Index: {PERSIST_DIR}</p>",
    unsafe_allow_html=True,
)
