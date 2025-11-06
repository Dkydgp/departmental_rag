"""
app.py - Streamlit RAG Chat Interface
Interactive chat interface for querying indexed documents
"""

import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from chromadb import PersistentClient

load_dotenv()

# Configuration
PERSIST_DIR = os.getenv("PERSIST_DIR", "embeddings")

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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model info
    st.subheader("📡 Model Configuration")
    embed_model = os.getenv('OPENROUTER_EMBED_MODEL', 'Not set')
    llm_model = os.getenv('OPENROUTER_MODEL', 'Not set')
    st.info(f"""
    **Embedding:** {embed_model}
    
    **LLM:** {llm_model}
    """)
    
    # Chat settings
    st.subheader("💬 Chat Settings")
    chat_mode = st.selectbox(
        "Chat Mode",
        ["context", "condense_question", "simple"],
        index=0,
        help="Context mode uses retrieved context for each query"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    
    # Actions
    st.markdown("---")
    st.subheader("🔧 Actions")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.session_state.messages:
        # Export chat
        st.subheader("💾 Export")
        export_text = ""
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "Assistant"
            export_text += f"{role}: {msg['content']}\n\n"
        
        st.download_button(
            label="Download Chat",
            data=export_text,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Tips
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Ask specific questions
    - Reference topics or chapters
    - Use follow-up questions
    - Request examples or explanations
    """)
    
    # Stats
    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### 📊 Stats")
        st.metric("Messages", len(st.session_state.messages))

# Main content area
# Check if index exists
if not os.path.exists(PERSIST_DIR):
    st.error("❌ **No index found!**")
    st.info("""
    ### Getting Started:
    
    1. **Add PDFs** to your Supabase bucket
    2. **Run indexing:**
       ```bash
       python rag_llama.py --build
       ```
    3. **Return here** and refresh the page
    """)
    st.stop()

# Load index (only once)
if not st.session_state.index_loaded:
    try:
        with st.spinner("🔄 Loading index and initializing models..."):
            # Validate env vars
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
            
            # Initialize embedding model
            embed_model = OpenAIEmbedding(
                model=os.getenv("OPENROUTER_EMBED_MODEL"),
                api_base=os.getenv("OPENROUTER_BASE_URL"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
            
            # Load index
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(
                storage_context,
                embed_model=embed_model
            )
            
            # Initialize LLM
            llm = OpenAI(
                model=os.getenv("OPENROUTER_MODEL"),
                api_base=os.getenv("OPENROUTER_BASE_URL"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=temperature
            )
            
            # Create chat engine
            st.session_state.chat_engine = index.as_chat_engine(
                llm=llm,
                chat_mode=chat_mode,
                verbose=False
            )
            
            st.session_state.index_loaded = True
            st.success("✅ System ready!")
            
    except Exception as e:
        st.error(f"❌ **Error loading index:** {str(e)}")
        st.info("Try rebuilding the index: `python rag_llama.py --rebuild`")
        st.stop()

# Display chat history
if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    # Welcome message
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Welcome!** I'm your study assistant.
        
        Ask me anything about your study materials:
        - Explain concepts
        - Summarize topics
        - Answer specific questions
        - Provide examples
        
        *What would you like to know?*
        """)

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Thinking..."):
                # Get response from chat engine
                response = st.session_state.chat_engine.chat(prompt)
                answer = response.response
                
                # Display response
                message_placeholder.markdown(answer)
                
                # Show sources if available
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    with st.expander("📚 View Sources"):
                        for i, node in enumerate(response.source_nodes, 1):
                            score = getattr(node, 'score', 0)
                            st.markdown(f"**Source {i}** (Relevance: {score:.3f})")
                            st.text(node.text[:300] + "..." if len(node.text) > 300 else node.text)
                            st.markdown("---")
            
            # Add to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
            
        except Exception as e:
            error_msg = f"❌ **Error:** {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
            st.error("Try rephrasing your question or check your API configuration.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.9em;'>"
    "Powered by LlamaIndex + OpenRouter | "
    f"Index Location: {PERSIST_DIR}"
    "</p>",
    unsafe_allow_html=True
)
