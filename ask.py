import os

from openai import OpenAI

from indexer import collection, embed, QUERY_PREFIX, OLLAMA_URL
from parse_pdf import count_tokens
from retrieval import Retriever

TOP_K = 5
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
MAX_DISTANCE = 0.85
RULES = "Answer only from the context. If the context contains the answer, give it directly. If it does not, say you don't know."

llm = OpenAI(base_url=OLLAMA_URL, api_key="ollama")


# Finds the chunks. Plain vector search unless HYBRID / RERANK / REWRITE are set (see retrieval.py).
retriever = Retriever(collection, lambda text: embed([text], QUERY_PREFIX)[0], llm, LLM_MODEL)


def shares_phrase(answer_text, chunk_text, size=6):
    words = answer_text.lower().split()
    chunk = " ".join(chunk_text.lower().split())
    for i in range(len(words) - size + 1):
        if " ".join(words[i:i + size]) in chunk:
            return True
    return False


def answer(question, history=None):
    chunks, best = retriever.retrieve(question, TOP_K, history)

    if best is not None:
        print("Best distance:", round(best, 2))
    if len(retriever.last_queries) > 1:
        print("Search query:", retriever.last_queries[-1])

    if not chunks or best > MAX_DISTANCE:
        return "I could not find this in your documents.", []

    context = "\n\n".join(
        f"[{c['source']}, page {c['page']}]\n{c['text']}" for c in chunks
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\n{RULES}"
    print("Prompt tokens (approx):", count_tokens(prompt))

    reply = llm.chat.completions.create(
        model=LLM_MODEL,
        temperature= 0,
        messages=[
            {"role": "system", "content": RULES},
            {"role": "user", "content": prompt},
        ],
    )
    text = reply.choices[0].message.content.strip()
    if text.lower().startswith(("i could not find", "don't know", "i don't know", "i do not know")):
        return text, []

    cited = []
    for chunk in chunks:
        if shares_phrase(text, chunk["text"]):
            label = (chunk["source"], chunk["page"])
            if label not in cited:
                cited.append(label)
    if not cited:
        cited = [(chunks[0]["source"], chunks[0]["page"])]
    return text, cited

if __name__ == "__main__":
    import sys
    import time

    question = " ".join(sys.argv[1:])
    start = time.time()
    text, sources = answer(question)
    print(text)
    for source, page in sources:
        print(f"From: {source}, page {page}")
    print("Seconds:", round(time.time() - start, 1))