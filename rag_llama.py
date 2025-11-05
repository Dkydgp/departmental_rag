import os, tempfile, requests
from supabase import create_client
from dotenv import load_dotenv
from tqdm import tqdm
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from chromadb import PersistentClient

load_dotenv()

# Supabase setup
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
BUCKET = os.getenv("SUPABASE_BUCKET", "books")

# Local persistent Chroma storage
persist_dir = os.getenv("PERSIST_DIR", "embeddings")
chroma_client = PersistentClient(path=persist_dir)

def fetch_pdfs():
    files = supabase.storage.from_(BUCKET).list()
    urls = [f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/{BUCKET}/{f['name']}"
            for f in files if f["name"].endswith(".pdf")]
    os.makedirs("tmp_pdfs", exist_ok=True)
    paths = []
    for url in tqdm(urls, desc="Downloading PDFs"):
        r = requests.get(url)
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir="tmp_pdfs")
        t.write(r.content)
        t.close()
        paths.append(t.name)
    return paths

def build_index():
    paths = fetch_pdfs()
    docs = SimpleDirectoryReader(input_files=paths).load_data()
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(docs)
    embed_model = OpenAIEmbedding(
        model=os.getenv("OPENROUTER_EMBED_MODEL"),
        api_base=os.getenv("OPENROUTER_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    vector_store = ChromaVectorStore(chroma_client=chroma_client, collection_name="books")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    index.storage_context.persist(persist_dir)
    print("✅ Index built and saved in", persist_dir)

if __name__ == "__main__":
    build_index()
