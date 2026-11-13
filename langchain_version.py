"""DocuChat built with LangChain: the same pipeline, using LangChain's ready-made parts.

    pip install -r requirements-langchain.txt
    python langchain_version.py sample_docs\\attention.pdf "How many attention heads does the base model use?"

This is a second, smaller version for comparison with the hand-built one. It has its own
index (`chroma_lc/`) and does not touch `chroma_db/`. Mapping to the hand-built code:

    PyPDFLoader                    <->  parse_pdf.py reading pages
    RecursiveCharacterTextSplitter <->  parse_pdf.py split_into_chunks()
    Chroma (langchain_chroma)      <->  indexer.py + search
    prompt | ChatOllama | parser   <->  ask.py answer()
"""

import os
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")   # native address, no /v1
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DB_PATH = os.getenv("LC_DB_PATH", "chroma_lc")
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
TOP_K = 5
MAX_DISTANCE = 0.85
RULES = "Answer only from the context. If the context contains the answer, give it directly. If it does not, say you don't know."


class PrefixedOllamaEmbeddings(OllamaEmbeddings):
    """OllamaEmbeddings does not add the nomic task prefixes, so we add them here."""

    def embed_documents(self, texts):
        return super().embed_documents([DOC_PREFIX + text for text in texts])

    def embed_query(self, text):
        return super().embed_query(QUERY_PREFIX + text)


def make_store():
    embeddings = PrefixedOllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)
    return Chroma(collection_name="docuchat_lc", embedding_function=embeddings,
                  persist_directory=DB_PATH)


# ---------- 1. load -> 2. split -> 3. store ----------
def index_pdf(store, path):
    name = Path(path).name
    pages = PyPDFLoader(str(path)).load()                  # one Document per page
    for page in pages:
        page.metadata = {"source": name, "page": page.metadata["page"] + 1}   # loader counts from 0

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=500, chunk_overlap=75
    )                                                      # counts tokens, like parse_pdf.py
    chunks = splitter.split_documents(pages)

    old = store.get(where={"source": name})["ids"]         # re-upload replaces the old chunks
    if old:
        store.delete(ids=old)
    store.add_documents(chunks)
    return len(chunks)


# ---------- 4. the chain ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", RULES),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\n" + RULES),
])


def shares_phrase(answer_text, chunk_text, size=6):
    words = answer_text.lower().split()
    chunk = " ".join(chunk_text.lower().split())
    return any(" ".join(words[i:i + size]) in chunk for i in range(len(words) - size + 1))


def ask(store, question):
    hits = store.similarity_search_with_score(question, k=TOP_K)   # (Document, distance)
    if not hits or hits[0][1] > MAX_DISTANCE:
        return "I could not find this in your documents.", []

    context = "\n\n".join(
        f"[{doc.metadata['source']}, page {doc.metadata['page']}]\n{doc.page_content}"
        for doc, _ in hits
    )
    chain = prompt | ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE, temperature=0) | StrOutputParser()
    text = chain.invoke({"context": context, "question": question}).strip()
    if text.lower().startswith(("i could not find", "don't know", "i don't know", "i do not know")):
        return text, []

    cited = []
    for doc, _ in hits:
        label = (doc.metadata["source"], doc.metadata["page"])
        if shares_phrase(text, doc.page_content) and label not in cited:
            cited.append(label)
    if not cited:
        cited = [(hits[0][0].metadata["source"], hits[0][0].metadata["page"])]
    return text, cited


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit('Usage: python langchain_version.py <pdf> "<question>"')
    store = make_store()
    pdf = Path(sys.argv[1])
    if not store.get(where={"source": pdf.name}, limit=1)["ids"]:
        print("Indexing", pdf.name, "...")
        print("Chunks:", index_pdf(store, pdf))
    answer_text, sources = ask(store, " ".join(sys.argv[2:]))
    print(answer_text)
    for source, page in sources:
        print(f"From: {source}, page {page}")
