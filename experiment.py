"""Retrieval experiment: which chunk size and embedding model find the right page?

Part A: 3 chunk sizes x 2 embedding models, 20 questions, plain vector search.
Part B: on one setup, compare dense search, hybrid, re-rank and query rewriting.

It tests RETRIEVAL only (does a top-5 chunk come from the right page?). It never calls the
answer model, so Part A is much faster than eval.py. It builds its own index in `chroma_exp/`
and never touches your real `chroma_db/`.

    ollama pull bge-m3          (once, about 1.2 GB)
    python experiment.py        (writes eval/experiment.csv and fills the README)

Options:  --sizes 500  --models nomic-embed-text  --skip-modes
If it stops half way, run it again: finished PDFs are not indexed twice.
"""

import argparse
import csv
import datetime
import json
import os
import re
import time
from pathlib import Path

import chromadb

import parse_pdf
from indexer import ollama
from questions import EXTRA_QUESTIONS, QUESTIONS
from retrieval import Retriever

DOCS_DIR = Path("sample_docs")
EXP_DB = os.getenv("EXP_DB_PATH", "chroma_exp")
OUT_DIR = Path("eval")
README = Path("README.md")
TOP_K = 5
BATCH = 16
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")

# model name -> (prefix for chunks, prefix for questions)
# nomic needs prefixes (model card). bge-m3: no prefix is used (assumption, not tested here).
MODELS = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "bge-m3": ("", ""),
}
SIZES = [200, 500, 800]          # chunk size in tokens; overlap is 15% of it (500 -> 75)
DEFAULT_SETUP = ("nomic-embed-text", 500)   # the setup DocuChat uses

ANSWERABLE = [q for q in QUESTIONS if not q.get("refuse")] + EXTRA_QUESTIONS
TRAPS = [q for q in QUESTIONS if q.get("refuse")]


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def find_pdfs():
    found = {}
    for name in sorted({q["source"] for q in ANSWERABLE}):
        matches = list(DOCS_DIR.rglob(name))
        if matches:
            found[name] = matches[0]
        else:
            print("Missing PDF, its questions are skipped:", name)
    return found


def embed(model, texts, prefix):
    vectors = []
    for start in range(0, len(texts), BATCH):
        batch = [prefix + text for text in texts[start:start + BATCH]]
        reply = ollama.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in reply.data)
    return vectors


# ---------- build one index per (model, chunk size) ----------
def load_times():
    path = Path(EXP_DB) / "times.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_times(times):
    (Path(EXP_DB) / "times.json").write_text(json.dumps(times, indent=2))


def build(client, model, size, pdfs, times):
    name = f"exp_{slug(model)}_{size}"
    collection = client.get_or_create_collection(name)
    done = {meta["source"] for meta in collection.get(include=["metadatas"])["metadatas"]}
    doc_prefix = MODELS[model][0]

    parse_pdf.CHUNK_TOKENS = size
    parse_pdf.OVERLAP_TOKENS = round(size * 0.15)
    for pdf_name, path in pdfs.items():
        if pdf_name in done:
            continue
        start = time.time()
        records = parse_pdf.parse_pdf(path)
        vectors = embed(model, [r["text"] for r in records], doc_prefix)
        collection.add(
            ids=[f"{pdf_name}:{i}" for i in range(len(records))],
            documents=[r["text"] for r in records],
            embeddings=vectors,
            metadatas=[{"source": r["source"], "page": r["page"]} for r in records],
        )
        seconds = round(time.time() - start, 1)
        times[name] = round(times.get(name, 0) + seconds, 1)
        save_times(times)
        print(f"  {name}: {pdf_name} -> {len(records)} chunks in {seconds} s")
    return collection


# ---------- scoring ----------
def first_hit_rank(item, chunks):
    """1-based rank of the first chunk from the right file AND a right page, else None."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk["source"] == item["source"] and chunk["page"] in item["pages"]:
            return rank
    return None


def score(ranks):
    count = len(ranks)
    hit1 = sum(1 for r in ranks if r == 1)
    hit5 = sum(1 for r in ranks if r is not None)
    mrr = sum(1 / r for r in ranks if r is not None) / count if count else 0
    return hit1, hit5, round(mrr, 2), count


def run_questions(retriever, questions):
    ranks, seconds = [], 0.0
    for item in questions:
        start = time.time()
        chunks, _ = retriever.retrieve(item["question"], TOP_K)
        seconds += time.time() - start
        ranks.append(first_hit_rank(item, chunks))
    return ranks, seconds / max(len(questions), 1)


def distances(retriever, questions):
    values = []
    for item in questions:
        _, best = retriever.retrieve(item["question"], 1)
        values.append(best)
    return [v for v in values if v is not None]


# ---------- Part A ----------
def part_a(client, models, sizes, pdfs, times):
    questions = [q for q in ANSWERABLE if q["source"] in pdfs]
    traps = TRAPS
    rows = []
    for model in models:
        for size in sizes:
            print(f"Part A: {model}, {size} tokens")
            collection = build(client, model, size, pdfs, times)
            query_prefix = MODELS[model][1]
            retriever = Retriever(
                collection, lambda text, m=model, p=query_prefix: embed(m, [text], p)[0],
                hybrid=False, rerank=False, rewrite=False,
            )
            ranks, seconds = run_questions(retriever, questions)
            hit1, hit5, mrr, count = score(ranks)
            real = distances(retriever, questions)
            trap = distances(retriever, traps)
            rows.append({
                "embedding_model": model, "chunk_tokens": size, "chunks": collection.count(),
                "questions": count, "hit_at_1": hit1, "hit_at_5": hit5, "mrr": mrr,
                "worst_real_distance": round(max(real), 2) if real else "",
                "closest_trap_distance": round(min(trap), 2) if trap else "",
                "index_seconds": times.get(f"exp_{slug(model)}_{size}", ""),
            })
            print(f"  hit@1 {hit1}/{count}  hit@5 {hit5}/{count}  MRR {mrr}")
    return rows


# ---------- Part B ----------
MODES = [
    ("dense (what DocuChat does now)", dict(hybrid=False, rerank=False, rewrite=False)),
    ("hybrid (vector + keyword)", dict(hybrid=True, rerank=False, rewrite=False)),
    ("dense + re-rank", dict(hybrid=False, rerank=True, rewrite=False)),
    ("hybrid + re-rank", dict(hybrid=True, rerank=True, rewrite=False)),
    ("hybrid + query rewriting", dict(hybrid=True, rerank=False, rewrite=True)),
]


def part_b(client, pdfs, times):
    model, size = DEFAULT_SETUP
    print(f"Part B: {model}, {size} tokens")
    collection = build(client, model, size, pdfs, times)
    query_prefix = MODELS[model][1]
    questions = [q for q in ANSWERABLE if q["source"] in pdfs]

    rows = []
    for label, flags in MODES:
        retriever = Retriever(
            collection, lambda text: embed(model, [text], query_prefix)[0],
            ollama, LLM_MODEL, **flags,
        )
        try:
            ranks, seconds = run_questions(retriever, questions)
        except ImportError as error:
            print(f"  {label}: skipped ({error}). Install the missing package: "
                  "pip install rank-bm25 sentence-transformers")
            rows.append({"mode": label, "questions": len(questions), "hit_at_1": "",
                         "hit_at_5": "", "mrr": "", "seconds_per_question": "not run"})
            continue
        hit1, hit5, mrr, count = score(ranks)
        rows.append({"mode": label, "questions": count, "hit_at_1": hit1, "hit_at_5": hit5,
                     "mrr": mrr, "seconds_per_question": round(seconds, 2)})
        print(f"  {label}: hit@1 {hit1}/{count}  hit@5 {hit5}/{count}  MRR {mrr}  {round(seconds, 2)} s")
    return rows


# ---------- output ----------
def save_csv(path, rows):
    if not rows:
        return
    OUT_DIR.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_block(rows_a, rows_b):
    lines = [
        "<!-- EXPERIMENT:START -->",
        f"Run on {datetime.date.today().isoformat()} with `python experiment.py`. 20 questions, "
        f"top-{TOP_K} chunks. **Hit@5** = a top-5 chunk comes from the right file and a right "
        "page. **Hit@1** = the first chunk does. **MRR** (Mean Reciprocal Rank) = average of "
        "1/rank of the first right chunk, so 1.0 is perfect. Only retrieval is tested, not "
        "the answer text.",
        "",
    ]
    if rows_a:
        lines += [
            "**Chunk size x embedding model** (plain vector search)",
            "",
            "| Embedding model | Chunk size | Chunks | Hit@1 | Hit@5 | MRR | Worst real question distance | Closest off-topic distance | Index time (s) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows_a:
            lines.append(
                f"| {r['embedding_model']} | {r['chunk_tokens']} | {r['chunks']} | "
                f"{r['hit_at_1']}/{r['questions']} | {r['hit_at_5']}/{r['questions']} | {r['mrr']} | "
                f"{r['worst_real_distance']} | {r['closest_trap_distance']} | {r['index_seconds']} |"
            )
        lines += [
            "",
            "The two distance columns decide the refusal gate: it only works for a model if the "
            "worst real question is closer than the closest off-topic one. Distances from "
            "different models are not comparable with each other.",
            "",
        ]
    if rows_b:
        model, size = DEFAULT_SETUP
        lines += [
            f"**Retrieval upgrades** (on `{model}`, {size}-token chunks)",
            "",
            "| Retrieval | Hit@1 | Hit@5 | MRR | Seconds per question |",
            "|---|---|---|---|---|",
        ]
        for r in rows_b:
            hit1 = f"{r['hit_at_1']}/{r['questions']}" if r["hit_at_1"] != "" else "-"
            hit5 = f"{r['hit_at_5']}/{r['questions']}" if r["hit_at_5"] != "" else "-"
            lines.append(f"| {r['mode']} | {hit1} | {hit5} | {'-' if r['mrr'] == '' else r['mrr']} | {r['seconds_per_question']} |")
        lines.append("")
    lines.append("<!-- EXPERIMENT:END -->")
    return "\n".join(lines)


def update_readme(block):
    if not README.exists():
        print("README.md not found, results are only in the eval folder")
        return
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(r"<!-- EXPERIMENT:START -->.*?<!-- EXPERIMENT:END -->", re.S)
    if not pattern.search(text):
        print("No EXPERIMENT markers in README.md, results are only in the eval folder")
        return
    README.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")
    print("README.md updated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--skip-modes", action="store_true", help="skip Part B")
    args = parser.parse_args()

    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit("No test PDFs found in sample_docs/. See the README for the list.")

    Path(EXP_DB).mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=EXP_DB)
    times = load_times()

    rows_a = part_a(client, args.models, args.sizes, pdfs, times)
    rows_b = [] if args.skip_modes else part_b(client, pdfs, times)

    save_csv(OUT_DIR / "experiment.csv", rows_a)
    save_csv(OUT_DIR / "modes.csv", rows_b)
    block = build_block(rows_a, rows_b)
    update_readme(block)
    print()
    print(block)


if __name__ == "__main__":
    main()
