# ingest.py
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = os.getenv("PERSIST_DIR", "embeddings")
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

def load_pdfs(data_dir):
    docs = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(data_dir, fname)
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()  # returns pages with metadata
            # add filename metadata
            for p in pages:
                p.metadata["source_file"] = fname
            docs.extend(pages)
    return docs

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def create_embeddings(texts):
    if USE_OPENAI:
        emb = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"),
                               model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    else:
        # fallback: sentence-transformers (offline)
        from langchain.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embedding=emb, persist_directory=PERSIST_DIR)
    db.persist()
    return db

def main():
    print("Loading PDFs...")
    raw_docs = load_pdfs(DATA_DIR)
    print(f"Loaded {len(raw_docs)} pages.")
    print("Chunking...")
    chunks = chunk_documents(raw_docs, chunk_size=800, chunk_overlap=150)
    print(f"Created {len(chunks)} chunks.")
    print("Creating embeddings & saving to Chroma...")
    db = create_embeddings(chunks)
    print("Done. Embeddings saved to:", PERSIST_DIR)

if __name__ == "__main__":
    main()
