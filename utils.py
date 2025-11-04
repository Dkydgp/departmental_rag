# utils.py
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
PERSIST_DIR = os.getenv("PERSIST_DIR", "embeddings")
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

def get_retriever(k=4):
    if USE_OPENAI:
        emb = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"),
                               model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    else:
        from langchain.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
    return db.as_retriever(search_kwargs={"k": k})

def make_qa_chain(retriever, temperature=0.2, model_name=None):
    if not model_name:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa
