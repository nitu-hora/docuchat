"""Finding the chunks for a question.

Plain search (default): embed the question, ask Chroma for the closest chunks.

Three optional upgrades, each switched on by an environment setting:
  HYBRID=1   also run keyword search (BM25) and merge the two result lists
  RERANK=1   re-score the merged candidates with a cross-encoder
  REWRITE=1  ask the LLM to rewrite the question as a better search query
"""

import os
import re
import unicodedata

CANDIDATES = int(os.getenv("CANDIDATES", "20"))   # how many chunks each search hands on
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RRF_K = 60                                        # standard value for rank merging

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with",
}


def flag(name):
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes", "on")


def tokenize(text):
    """Lowercase words for keyword search. NFKC turns PDF ligature characters (one-character "fi", "fl") into plain letters."""
    text = unicodedata.normalize("NFKC", text).lower()
    words = re.findall(r"\d+(?:[.,]\d+)*|[^\W_]+", text)
    return [word for word in words if word not in STOP_WORDS]


def fuse(result_lists):
    """Reciprocal Rank Fusion (RRF): a chunk scores 1/(60+rank) in every list it appears in."""
    scores = {}
    chunks = {}
    for result in result_lists:
        for rank, chunk in enumerate(result, start=1):
            scores[chunk["id"]] = scores.get(chunk["id"], 0) + 1 / (RRF_K + rank)
            first = chunks.setdefault(chunk["id"], dict(chunk))
            if first.get("distance") is None and chunk.get("distance") is not None:
                first["distance"] = chunk["distance"]
    order = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [chunks[chunk_id] for chunk_id in order]


class Retriever:
    def __init__(self, collection, embed_query, llm=None, llm_model=None,
                 hybrid=None, rerank=None, rewrite=None):
        self.collection = collection
        self.embed_query = embed_query      # function: text -> vector
        self.llm = llm                      # OpenAI-style client, only used to rewrite
        self.llm_model = llm_model
        self.hybrid = flag("HYBRID") if hybrid is None else hybrid
        self.rerank = flag("RERANK") if rerank is None else rerank
        self.rewrite = flag("REWRITE") if rewrite is None else rewrite
        self.last_queries = []
        self._reranker = None
        self._bm25 = None
        self._bm25_key = None
        self._rows = []

    def mode(self):
        parts = ["hybrid" if self.hybrid else "dense"]
        if self.rerank:
            parts.append("rerank")
        if self.rewrite:
            parts.append("rewrite")
        return "+".join(parts)

    # ---------- the main call ----------
    def retrieve(self, question, top_k=5, history=None):
        """Return (chunks, best_distance). best_distance is the closest vector distance,
        used by the caller as the 'is this question about my documents?' gate."""
        queries = [question]
        if self.rewrite:
            rewritten = self._rewrite(question, history)
            if rewritten.lower() != question.lower():
                queries.append(rewritten)
        self.last_queries = queries

        lists = []
        best = None
        for query in queries:
            dense = self._dense(query, CANDIDATES)
            if dense:
                best = dense[0]["distance"] if best is None else min(best, dense[0]["distance"])
            lists.append(dense)
            if self.hybrid:
                lists.append(self._keyword(query, CANDIDATES))

        lists = [result for result in lists if result]
        if not lists:
            return [], None

        pool = lists[0] if len(lists) == 1 else fuse(lists)
        if self.rerank:
            return self._rerank(queries[-1], pool[:CANDIDATES], top_k), best
        return pool[:top_k], best

    # ---------- vector (meaning) search ----------
    def _dense(self, query, n):
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[self.embed_query(query)], n_results=min(n, count)
        )
        return [
            {"id": chunk_id, "text": text, "source": meta["source"], "page": meta["page"],
             "distance": distance}
            for chunk_id, text, meta, distance in zip(
                result["ids"][0], result["documents"][0],
                result["metadatas"][0], result["distances"][0],
            )
        ]

    # ---------- keyword (BM25) search ----------
    def _load_bm25(self):
        data = self.collection.get(include=["documents", "metadatas"])
        key = (len(data["ids"]), hash(tuple(data["documents"])))
        if key != self._bm25_key:
            from rank_bm25 import BM25Okapi  # imported here so plain mode needs no extra package

            self._rows = [
                {"id": chunk_id, "text": text, "source": meta["source"], "page": meta["page"],
                 "distance": None}
                for chunk_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
            ]
            tokens = [tokenize(row["text"]) for row in self._rows]
            self._bm25 = BM25Okapi(tokens) if any(tokens) else None
            self._bm25_key = key
        return self._bm25

    def _keyword(self, query, n):
        bm25 = self._load_bm25()
        words = tokenize(query)
        if bm25 is None or not words:
            return []
        scores = bm25.get_scores(words)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [dict(self._rows[i]) for i in order if scores[i] > 0]

    # ---------- re-ranker ----------
    def _rerank(self, query, chunks, top_k):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder  # heavy, loaded only when asked

            self._reranker = CrossEncoder(RERANK_MODEL)
        scores = self._reranker.predict([(query, chunk["text"]) for chunk in chunks])
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in order[:top_k]]

    # ---------- query rewriting ----------
    def _rewrite(self, question, history):
        if self.llm is None:
            return question

        chat = []
        for message in (history or [])[-4:]:
            text = str(message.get("content", "")).split("\n\nFrom:")[0].strip()
            role = "User" if message.get("role") == "user" else "Assistant"
            chat.append(f"{role}: {text[:300]}")

        if chat:
            prompt = (
                "Chat so far:\n" + "\n".join(chat) + f"\n\nLatest question: {question}\n\n"
                "Rewrite the latest question so it makes sense on its own: replace words like "
                "'it' or 'that' using the chat. Keep names, numbers and technical terms. "
                "Output only the rewritten question."
            )
        else:
            prompt = (
                f"Question: {question}\n\n"
                "Rewrite it as a short search query with its key terms and names. "
                "Output only the query."
            )
        reply = self.llm.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            max_tokens=60,
            messages=[
                {"role": "system",
                 "content": "You rewrite questions for a search engine over documents. "
                            "You never answer the question."},
                {"role": "user", "content": prompt},
            ],
        )
        lines = (reply.choices[0].message.content or "").strip().strip('"').splitlines()
        text = lines[0].strip() if lines else ""
        return text if 0 < len(text) <= 300 else question
