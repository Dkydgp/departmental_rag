"""
rag_llama.py - RAG Indexing Script
Downloads PDFs from Supabase bucket "Books" and builds vector embeddings
"""

import os
import tempfile
import shutil
import hashlib
import json
import argparse
from datetime import datetime
from urllib.parse import quote
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from chromadb import PersistentClient

# ============================================================
# 🔧 Load environment variables
# ============================================================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET = os.getenv("SUPABASE_BUCKET", "books").strip().lower()
PERSIST_DIR = os.getenv("PERSIST_DIR", "storage")
TMP_DIR = "tmp_pdfs"
METADATA_FILE = os.path.join(PERSIST_DIR, "index_metadata.json")


# ============================================================
# ✅ Validate environment
# ============================================================
def validate_environment():
    required = [
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_EMBED_MODEL",
        "OPENROUTER_BASE_URL",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
    print("✅ Environment validated")
    return True


# ============================================================
# 🧠 Metadata utilities
# ============================================================
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {"files": {}, "last_updated": None, "total_documents": 0, "total_nodes": 0}


def save_metadata(metadata):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    metadata["last_updated"] = datetime.now().isoformat()
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


# ============================================================
# 📥 Fetch PDFs from Supabase
# ============================================================
def fetch_pdfs(force_download=False):
    print(f"\n📥 Fetching PDFs from Supabase bucket: {BUCKET}")

    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        files = supabase.storage.from_(BUCKET).list()
        pdf_files = [f["name"] for f in files if f["name"].endswith(".pdf")]

        if not pdf_files:
            print("⚠️ No PDFs found in Supabase bucket.")
            return []

        base_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}"
        os.makedirs(TMP_DIR, exist_ok=True)

        metadata = load_metadata()
        downloaded, skipped = [], 0

        for filename in tqdm(pdf_files, desc="⬇️ Downloading PDFs"):
            encoded = quote(filename)
            url = f"{base_url}/{encoded}"

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.content
                file_hash = hashlib.md5(data).hexdigest()

                if not force_download and filename in metadata["files"]:
                    if metadata["files"][filename].get("hash") == file_hash:
                        skipped += 1
                        continue

                local_path = os.path.join(TMP_DIR, filename)
                with open(local_path, "wb") as f:
                    f.write(data)
                downloaded.append((local_path, filename, file_hash))

            except Exception as e:
                print(f"⚠️ Error downloading {filename}: {e}")

        print(f"\n✅ Downloaded: {len(downloaded)} | ⏩ Skipped: {skipped}")
        return downloaded

    except Exception as e:
        print(f"❌ Error fetching PDFs: {e}")
        return []


# ============================================================
# ⚡ Build the vector index
# ============================================================
def build_index(force_rebuild=False):
    print("\n" + "=" * 60)
    print("🚀 Building RAG Index")
    print("=" * 60)

    if not validate_environment():
        return False

    pdf_data = fetch_pdfs(force_download=force_rebuild)
    if not pdf_data:
        print("\n⚠️ No PDFs to process.")
        return True

    try:
        pdf_paths = [path for path, _, _ in pdf_data]
        print(f"\n📖 Loading {len(pdf_paths)} document(s)...")
        docs = SimpleDirectoryReader(input_files=pdf_paths).load_data()
        print(f"✅ Loaded {len(docs)} document(s)")

        print("🔪 Splitting into chunks...")
        parser = SimpleNodeParser.from_defaults(chunk_size=800, chunk_overlap=150)
        nodes = parser.get_nodes_from_documents(docs, show_progress=True)
        print(f"✅ Created {len(nodes)} nodes")
        print(f"🔤 Embedding {len(nodes)} nodes using {os.getenv('OPENROUTER_EMBED_MODEL')}")

        embed_model = OpenAIEmbedding(
            model=os.getenv("OPENROUTER_EMBED_MODEL", "text-embedding-3-small"),
            api_base=os.getenv("OPENROUTER_BASE_URL"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        print("💾 Setting up Chroma vector store...")
        os.makedirs(PERSIST_DIR, exist_ok=True)
        chroma_client = PersistentClient(path=PERSIST_DIR)

        if force_rebuild:
            try:
                chroma_client.delete_collection("Books")
            except Exception:
                shutil.rmtree(PERSIST_DIR, ignore_errors=True)

        vector_store = ChromaVectorStore(chroma_client=chroma_client, collection_name="Books")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print("⚡ Building vector index...")
        index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
        index.storage_context.persist(persist_dir=PERSIST_DIR)

        metadata = load_metadata()
        for _, filename, file_hash in pdf_data:
            metadata["files"][filename] = {
                "hash": file_hash,
                "processed": datetime.now().isoformat(),
            }
        metadata["total_documents"] = len(docs)
        metadata["total_nodes"] = len(nodes)
        save_metadata(metadata)

        print("\n✅ SUCCESS! RAG Index Built")
        print(f"📊 Documents: {len(docs)} | Nodes: {len(nodes)}")
        print(f"📁 Saved to: {PERSIST_DIR}\n")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR)
            print("🧹 Cleaned up temporary files")


# ============================================================
# 📊 Show index status
# ============================================================
def show_status():
    print("\n" + "=" * 60)
    print("📊 Index Status")
    print("=" * 60)

    if not os.path.exists(PERSIST_DIR):
        print("❌ No index found")
        print("Run: python rag_llama.py --build")
        return

    metadata = load_metadata()
    print(f"📁 Location: {PERSIST_DIR}")
    print(f"📄 Documents: {metadata.get('total_documents', 0)}")
    print(f"📦 Nodes: {metadata.get('total_nodes', 0)}")
    print(f"🕐 Last Updated: {metadata.get('last_updated', 'Never')}")
    print("\n📚 Files indexed:")
    for filename, info in metadata.get("files", {}).items():
        print(f"  - {filename} ({info.get('processed', 'unknown')})")
    print("=" * 60 + "\n")


# ============================================================
# 🚀 Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Index Builder")
    parser.add_argument("--build", action="store_true", help="Build/update index")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild from scratch")
    parser.add_argument("--status", action="store_true", help="Show index status")
    args = parser.parse_args()

    if args.rebuild:
        build_index(force_rebuild=True)
    elif args.build:
        build_index(force_rebuild=False)
    elif args.status:
        show_status()
    else:
        if not os.path.exists(PERSIST_DIR):
            build_index()
        else:
            show_status()
            print("Tip: Use --build to update or --rebuild to start fresh")
