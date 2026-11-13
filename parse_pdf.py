#------------- settings and the token counter ------------
import re
from pathlib import Path
import tiktoken
from pypdf import PdfReader

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 75
MIN_PAGE_TOKEN = 30

encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(encoder.encode(text))


#-------------- cut one page's text into chunks ---------------
def split_into_chunks(text):
    text = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if current and current_tokens + sentence_tokens > CHUNK_TOKENS:
            chunks.append(" ".join(current))

            overlap = []
            overlap_tokens = 0
            for old_sentence in reversed(current):
                old_tokens = count_tokens(old_sentence)
                if overlap_tokens + old_tokens > OVERLAP_TOKENS:
                    break
                overlap.insert(0, old_sentence)
                overlap_tokens += old_tokens
            current = overlap
            current_tokens = overlap_tokens

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks

#------------- read the pdf page by page ------------------
def parse_pdf(path):
    path = Path(path)
    reader = PdfReader(path)

    records = []
    for page_number, page in enumerate(reader.pages, start = 1):
        text = page.extract_text() or ""
        if count_tokens(text)< MIN_PAGE_TOKEN:
            continue
        for piece in split_into_chunks(text):
            records.append({"text": piece, "source": path.name, "page": page_number})

    if not records:
        raise ValueError(f"no text found in {path.name}. It may be scanned PDF.")

    return records


#----------------------- test ---------------------------
if __name__ == "__main__":
    import sys

    records = parse_pdf(sys.argv[1])
    sizes = [count_tokens(r["text"]) for r in records]
    print("Smallest:", min(sizes), "tokens | Largest:", max(sizes), "tokens")
    print("--- first chunk ---")
    print(records[0]["source"], "page", records[0]["page"])
    print(records[0]["text"][:400])
