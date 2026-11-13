import csv
import datetime
import re
import statistics
import sys
import time
from pathlib import Path

from ask import LLM_MODEL, TOP_K, answer, retriever
from indexer import EMBED_MODEL, collection, index_pdf
from questions import QUESTIONS

DOCS_DIR = Path("sample_docs")
RESULTS_CSV = Path("eval") / "results.csv"
README = Path("README.md")



def is_indexed(name):
    return bool(collection.get(where={"source": name}, limit=1)["ids"])


def ensure_indexed(names):
    for name in names:
        if is_indexed(name):
            continue
        matches = list(DOCS_DIR.rglob(name))
        if not matches:
            print("Missing file, skipping:", name)
            continue
        print("Indexing", name, "...")
        start = time.time()
        count = index_pdf(matches[0])
        print(f"  {count} chunks in {round(time.time() - start)} s")


def mark(value):
    if value is None:
        return "-"
    return "pass" if value else "fail"


def run():
    sources = sorted({q["source"] for q in QUESTIONS if "source" in q})
    ensure_indexed(sources)

    rows = []
    for number, item in enumerate(QUESTIONS, start=1):
        refuse = item.get("refuse", False)
        if not refuse and not is_indexed(item["source"]):
            print(f"[{number}/{len(QUESTIONS)}] skipped (file not indexed): {item['question']}")
            continue

        start = time.time()
        text, cited = answer(item["question"])
        seconds = round(time.time() - start, 1)

        if refuse:
            answer_ok, page_ok = cited == [], None
        else:
            answer_ok = any(word in text.lower() for word in item["any_of"])
            page_ok = any(src == item["source"] and page in item["pages"] for src, page in cited)

        rows.append({
            "question": item["question"],
            "kind": "refuse" if refuse else "answer",
            "expected_source": item.get("source", ""),
            "expected_pages": " ".join(str(p) for p in item.get("pages", [])),
            "answer": text.replace("\n", " "),
            "cited": "; ".join(f"{s} p{p}" for s, p in cited),
            "answer_ok": answer_ok,
            "page_ok": page_ok,
            "seconds": seconds,
        })
        print(f"[{number}/{len(QUESTIONS)}] answer {mark(answer_ok)} | page {mark(page_ok)} | {seconds} s | {item['question']}")
    return rows


def save_csv(rows):
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_block(rows):
    answerable = [r for r in rows if r["kind"] == "answer"]
    refusals = [r for r in rows if r["kind"] == "refuse"]
    right_answer = sum(r["answer_ok"] for r in answerable)
    right_page = sum(bool(r["page_ok"]) for r in answerable)
    refused = sum(r["answer_ok"] for r in refusals)
    times = [r["seconds"] for r in answerable]
    typical = round(statistics.median(times)) if times else 0
    slowest = round(max(times)) if times else 0

    lines = [
        "<!-- EVAL:START -->",
        f"Run on {datetime.date.today().isoformat()} with `python eval.py`. Model `{LLM_MODEL}`, "
        f"embeddings `{EMBED_MODEL}`, retrieval `{retriever.mode()}`, top-{TOP_K} chunks, laptop CPU only. "
        "The answer check is automatic: the answer must contain a keyword that the PDF "
        "contains. It is not a human grade.",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Answer contains the expected fact | {right_answer}/{len(answerable)} |",
        f"| Cited page is a correct page | {right_page}/{len(answerable)} |",
        f"| Off-topic questions refused | {refused}/{len(refusals)} |",
        f"| Typical time per answered question (median) | {typical} s |",
        f"| Slowest question | {slowest} s |",
        "",
        "| Question | Answer | Page | Seconds |",
        "|---|---|---|---|",
    ]
    for r in rows:
        question = r["question"].replace("|", "/")
        lines.append(f"| {question} | {mark(r['answer_ok'])} | {mark(r['page_ok'])} | {r['seconds']} |")
    lines.append("")
    if times and slowest > 5 * typical:
        lines.append(
            f"The median is used because one question took {slowest} s, far above the rest. "
            "I did not investigate the cause. A single slow run would distort an average."
        )
    lines.append("<!-- EVAL:END -->")
    return "\n".join(lines)


def update_readme(block):
    if not README.exists():
        print("README.md not found, results are only in", RESULTS_CSV)
        return
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(r"<!-- EVAL:START -->.*?<!-- EVAL:END -->", re.S)
    if not pattern.search(text):
        print("No EVAL markers in README.md, results are only in", RESULTS_CSV)
        return
    README.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")
    print("README.md updated.")


def load_csv():
    """Read eval/results.csv back, so the README block can be rebuilt without re-running."""
    rows = []
    with open(RESULTS_CSV, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            row["answer_ok"] = row["answer_ok"] == "True"
            row["page_ok"] = None if row["page_ok"] == "" else row["page_ok"] == "True"
            row["seconds"] = float(row["seconds"])
            rows.append(row)
    return rows


if __name__ == "__main__":
    if "--report" in sys.argv:
        block = build_block(load_csv())
        update_readme(block)
        print(block)
        raise SystemExit
    results = run()
    if not results:
        raise SystemExit("No questions were run. Index the PDFs first.")
    save_csv(results)
    block = build_block(results)
    update_readme(block)
    print()
    print(block)
