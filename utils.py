# utils.py
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

def get_embeddings():
    """Embedding model from OpenRouter"""
    model = os.getenv("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

def get_openrouter_llm(temperature=0.2):
    """Chat LLM from OpenRouter"""
    model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

def get_retriever(k=4):
    embeddings = get_embeddings()
    db = Chroma(persist_directory=os.getenv("PERSIST_DIR", "embeddings"), embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": k})

def make_qa_chain(retriever, temperature=0.2):
    llm = get_openrouter_llm(temperature)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa
