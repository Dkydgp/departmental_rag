# ingest.py
import os
import tempfile
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from supabase import create_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "books")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")
PERSIST_DIR = os.getenv("PERSIST_DIR", "embeddings")

# Initialize Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def list_pdfs():
    """List all PDF files in your Supabase bucket"""
    files = supabase.storage.from_(SUPABASE_BUCKET).list()
    pdf_urls = [
        f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{f['name']}"
        for f in files if f["name"].endswith(".pdf")
    ]
    return pdf_urls

def download_pdf(url):
    """Download a PDF temporarily"""
    response = requests.get(url)
    response.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(response.content)
    tmp.close()
    return tmp.name

def load_pdfs_from_supabase():
    """Load all Supabase PDFs into LangChain Documents"""
    urls = list_pdfs()
    print(f"📚 Found {len(urls)} PDFs in Supabase bucket.")
    docs = []
    for url in tqdm(urls, desc="Downloading PDFs"):
        local_path = download_pdf(url)
        loader = PyPDFLoader(local_path)
        pages = loader.load_and_split()
        for p in pages:
            p.metadata["source_file"] = os.path.basename(url)
        docs.extend(pages)
    return docs

def chunk_documents(docs, chunk_size=800, overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

def create_vectorstore(chunks):
    print(f"🔹 Creating embeddings using {EMBED_MODEL} via OpenRouter...")
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL
    )
    db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    db.persist()
    print("✅ Embeddings stored in:", PERSIST_DIR)

def main():
    print("🚀 Starting ingestion from Supabase...")
    docs = load_pdfs_from_supabase()
    print(f"📄 Loaded {len(docs)} pages from PDFs.")
    chunks = chunk_documents(docs)
    print(f"🧩 Split into {len(chunks)} text chunks.")
    create_vectorstore(chunks)
    print("🎉 Ingestion complete. Ready to chat!")

if __name__ == "__main__":
    main()
