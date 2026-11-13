import os

import chromadb
from openai import OpenAI

from parse_pdf import parse_pdf

DB_PATH = os.getenv("DB_PATH", "chroma_db")
COLLECTION_NAME = "docuchat"
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1/")
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
BATCH_SIZE = 16

ollama = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
db = chromadb.PersistentClient(path=DB_PATH)
collection = db.get_or_create_collection(COLLECTION_NAME)

# ------------------ embed the function -------------------
def embed(texts, prefix):
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = [ prefix + text for text in texts[start:start +BATCH_SIZE]]
        reply = ollama.embeddings.create(model = EMBED_MODEL, input = batch)
        vectors.extend(item.embedding for item in reply.data)
    return vectors

#---------------- store one PDF ---------------------
def index_pdf(path):
    records = parse_pdf(path)
    source = records[0]["source"]

    collection.delete(where= {"source": source})

    collection.add(
        ids = [f"{source}:{number}" for number in range(len(records))],
        documents = [record["text"] for record in records],
        embeddings = embed([record["text"] for record in records], DOC_PREFIX),
        metadatas = [{"source": record["source"], "page": record["page"]} for record in records]
    )
    return len(records)



# ------------- test ------------------
if __name__ == "__main__":
    import sys

    added = index_pdf(sys.argv[1])
    print("Chunks added:", added)
    print("Total chunks in database:", collection.count())