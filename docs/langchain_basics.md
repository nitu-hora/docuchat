# LangChain basics, mapped to DocuChat

LangChain is a library of ready-made parts for the same steps you built by hand. You built
all of them, so each LangChain part below has a matching piece of your code.

**How I checked this.** I installed the packages and ran small checks (versions: `langchain` 1.5.0,
`langchain-core` 1.8.0, `langchain-text-splitters` 1.1.3, `langchain-community` 0.4.2,
`langchain-chroma` 1.1.0, `langchain-ollama` 1.1.0). I could not open LangChain's official docs
pages, so check names against the docs before you rely on them in an interview.

## The four parts your mentor listed

| Part | What it does | LangChain class | Your DocuChat code |
|---|---|---|---|
| Document loader | Reads a file into `Document` objects (text + metadata) | `PyPDFLoader` (in `langchain_community.document_loaders`, uses `pypdf`) | `parse_pdf()` reading pages with `PdfReader` |
| Text splitter | Cuts text into chunks | `RecursiveCharacterTextSplitter` (in `langchain_text_splitters`) | `split_into_chunks()` |
| Vector store | Saves chunks with their vectors and searches them | `Chroma` (in `langchain_chroma`) with `.add_documents()`, `.similarity_search()`, `.as_retriever()` | `collection.add()` and `search()` |
| Chain | Joins steps so the output of one feeds the next | `prompt \| llm \| StrOutputParser()` | `answer()`: search, gate, prompt, model, cite |

Two helpers you would also meet: `OllamaEmbeddings` (like your `embed()`) and `ChatOllama`
(like your `llm.chat.completions.create(...)`), both in `langchain_ollama`.

## What I saw when I ran them

- `PyPDFLoader` on `attention.pdf` gave **15 documents, one per page**. Its `page` number starts at **0**
  (your `parse_pdf` starts at 1, and `page_label` is the text "1"). If you mix the two you cite the wrong page.
- `RecursiveCharacterTextSplitter` counts **characters** by default. With 2,000 characters and 300 overlap
  it made **27 chunks** for the same paper (your token-based splitter made 29). Its default separators are
  paragraph break, line break, space, then single characters, which is why it is called "recursive": it tries
  the biggest separator first. Yours cuts at sentences.
- To count tokens instead, the splitter has `from_tiktoken_encoder(encoding_name=...)`. I did not run it,
  because `tiktoken` cannot download its data in my workspace.
- Chains today are built with `|`. `prompt | llm | StrOutputParser()` ran. The older helpers
  `create_retrieval_chain` and `RetrievalQA` are **not** in `langchain.chains` in version 1.5.0. They
  import from `langchain_classic.chains`.
- Importing `langchain_community` prints a notice that the package is being sunset, with a move to
  separate integration packages.

## DocuChat in LangChain: `langchain_version.py`

The same pipeline, written with the four parts above, is in `langchain_version.py` (about 100 lines).
Run it with:

```powershell
pip install -r requirements-langchain.txt
python langchain_version.py sample_docs\attention.pdf "How many attention heads does the base model use?"
```

It keeps its own index in `chroma_lc/`. What I changed compared with the plain sketch, and why:

- **`PrefixedOllamaEmbeddings`.** `OllamaEmbeddings` does not add the `search_document:` and `search_query:`
  prefixes (my reasoning; the test below confirmed the query side). A small subclass adds them.
- **Page numbers.** `PyPDFLoader` counts pages from 0, so the code adds 1 and uses the file name as `source`.
- **Token chunks.** `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)` counts tokens (500, overlap 75),
  like `parse_pdf.py`. It still cuts at paragraph, line and word breaks, not at sentences.
- **Gate and citations.** `similarity_search_with_score` returns the distance, so the same 0.85 gate works.
  The copied-phrase citation is copied over, because LangChain does not do citations for you.

**What was tested, and what was not.** It was first checked with a fake embedding function, a fake chat model and a fake
tokenizer. That confirmed: the PDF loads, pages are numbered from 1, re-indexing replaces old chunks, the
query gets its prefix, the chain returns text, the gate refuses, and a "don't know" answer shows no sources.
I then ran it with real Ollama on my laptop (`python langchain_version.py sample_docs\attention.pdf "How many attention heads does the base model use?"`).
It indexed `attention.pdf` into 29 chunks, answered "The base model uses 8 attention heads." and cited page 5.
I did not time it, so the seconds row is left empty.

| Same question, same PDF | Hand-built | LangChain version |
|---|---|---|
| Chunks for `attention.pdf` | 29 | 29 |
| Answer and cited page | 8 heads, page 5 (passed in `eval.py`) | 8 heads, page 5 |
| Seconds | not compared | not measured |

## What LangChain gives you and what it costs

| Gives | Costs |
|---|---|
| Less code, and parts you can swap (another loader, another vector store) | You no longer see the details: the distance gate, the citation logic and the prompt are yours to rebuild |
| The same pattern in every tutorial and job description | More packages, and names that change between versions (see the `langchain_classic` move) |

**Interview line:** "I built the pipeline by hand first, so I know what each LangChain class does
underneath: a loader reads pages, a splitter cuts chunks, a vector store saves and searches vectors,
and a chain passes one step's output to the next."
