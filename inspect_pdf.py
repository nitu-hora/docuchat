import sys
from pypdf import PdfReader

path = sys.argv[1]
reader = PdfReader(path)
print("Pages:", len(reader.pages))

for number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    print("page", number, "characters:", len(text))

print("--- page 1 text (first 1500 characters) ---")
print((reader.pages[0].extract_text() or "")[:1500])