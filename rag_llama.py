"""
rag_llama.py - RAG Indexing Script
Downloads PDFs from Supabase and builds vector embeddings
"""

import os
import tempfile
import requests
import shutil
import hashlib
import json
import argparse
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv
from tqdm import tqdm
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from chromadb import PersistentClient

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET = os.getenv("SUPABASE_BUCKET", "books")
PERSIST_DIR = os.getenv("PERSIST_DIR", "embeddings")
TMP_DIR = "tmp_pdfs"
METADATA_FILE = os.path.join(PERSIST_DIR, "index_metadata.json")


def validate_environment():
    """Validate required environment variables."""
    required = ["SUPABASE_URL", "SUPABASE_KEY", "OPENROUTER_API_KEY", 
                "OPENROUTER_EMBED_MODEL", "OPENROUTER_BASE_URL"]
    missing = [var for var in required if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
    
    print("✅ Environment validated")
    return True


def load_metadata():
    """Load existing index metadata."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {"files": {}, "last_updated": None, "total_documents": 0, "total_nodes": 0}


def save_metadata(metadata):
    """Save index metadata."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    metadata["last_updated"] = datetime.now().isoformat()
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def fetch_pdfs(force_download=False):
    """Download PDFs from Supabase."""
    print("\n📥 Fetching PDFs from Supabase...")
    
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        files = supabase.storage.from_(BUCKET).list()
        pdf_files = [f for f in files if f["name"].endswith(".pdf")]
        
        if not pdf_files:
            print("⚠️  No PDF files found")
            return []
        
        print(f"Found {len(pdf_files)} PDF(s)")
        os.makedirs(TMP_DIR, exist_ok=True)
        
        metadata = load_metadata()
        downloaded = []
        skipped = 0
        
        for file_info in tqdm(pdf_files, desc="Downloading"):
            filename = file_info["name"]
            url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{filename}"
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                file_hash = hashlib.md5(response.content).hexdigest()
                
                # Skip if unchanged
                if not force_download and filename in metadata["files"]:
                    if metadata["files"][filename].get("hash") == file_hash:
                        skipped += 1
                        continue
                
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf", dir=TMP_DIR
                )
                temp_file.write(response.content)
                temp_file.close()
                
                downloaded.append((temp_file.name, filename, file_hash))
                
            except Exception as e:
                print(f"\n⚠️  Error downloading {filename}: {e}")
        
        print(f"✅ Downloaded: {len(downloaded)} | Skipped: {skipped}")
        return downloaded
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def build_index(force_rebuild=False):
    """Build the vector index."""
    print("\n" + "="*60)
    print("🚀 Building RAG Index")
    print("="*60)
    
    if not validate_environment():
        return False
    
    pdf_data = fetch_pdfs(force_download=force_rebuild)
    
    if not pdf_data:
        print("\n⚠️  No PDFs to process")
        return True
    
    try:
        # Load documents
        pdf_paths = [path for path, _, _ in pdf_data]
        print(f"\n📖 Loading {len(pdf_paths)} document(s)...")
        docs = SimpleDirectoryReader(input_files=pdf_paths).load_data()
        print(f"✅ Loaded {len(docs)} document(s)")
        
        # Parse into nodes
        print("🔪 Parsing into chunks...")
        parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=200)
        nodes = parser.get_nodes_from_documents(docs, show_progress=True)
        print(f"✅ Created {len(nodes)} nodes")
        
        # Initialize embedding model
        print(f"🤖 Initializing embeddings: {os.getenv('OPENROUTER_EMBED_MODEL')}")
        embed_model = OpenAIEmbedding(
            model=os.getenv("OPENROUTER_EMBED_MODEL"),
            api_base=os.getenv("OPENROUTER_BASE_URL"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
        # Setup ChromaDB
        print(f"💾 Setting up ChromaDB: {PERSIST_DIR}")
        chroma_client = PersistentClient(path=PERSIST_DIR)
        
        if force_rebuild:
            try:
                chroma_client.delete_collection(name="books")
                print("🗑️  Deleted old collection")
            except:
                pass
        
        vector_store = ChromaVectorStore(
            chroma_client=chroma_client,
            collection_name="books"
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Build index
        print("⚡ Building vector index...")
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        # Persist
        print(f"💾 Saving to: {PERSIST_DIR}")
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        
        # Update metadata
        metadata = load_metadata()
        for _, filename, file_hash in pdf_data:
            metadata["files"][filename] = {
                "hash": file_hash,
                "processed": datetime.now().isoformat()
            }
        metadata["total_documents"] = len(docs)
        metadata["total_nodes"] = len(nodes)
        save_metadata(metadata)
        
        print("\n" + "="*60)
        print("✅ SUCCESS!")
        print(f"📊 Documents: {len(docs)} | Nodes: {len(nodes)}")
        print(f"📁 Location: {PERSIST_DIR}")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR)
            print("🧹 Cleaned up temp files")


def show_status():
    """Show current index status."""
    print("\n" + "="*60)
    print("📊 Index Status")
    print("="*60)
    
    if not os.path.exists(PERSIST_DIR):
        print("❌ No index found")
        print("Run: python rag_llama.py --build")
        return
    
    metadata = load_metadata()
    print(f"📁 Location: {PERSIST_DIR}")
    print(f"📄 Documents: {metadata.get('total_documents', 0)}")
    print(f"📦 Nodes: {metadata.get('total_nodes', 0)}")
    print(f"🕐 Last Updated: {metadata.get('last_updated', 'Never')}")
    print(f"\n📚 Files indexed:")
    for filename, info in metadata.get("files", {}).items():
        print(f"  - {filename} ({info.get('processed', 'unknown')})")
    print("="*60 + "\n")


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
        # Default: build if no index exists, otherwise show status
        if not os.path.exists(PERSIST_DIR):
            build_index()
        else:
            show_status()
            print("Tip: Use --build to update or --rebuild to start fresh")
