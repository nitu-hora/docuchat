import os
import shutil
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from ask import answer
from indexer import collection, index_pdf

# Uploaded PDFs are copied here so that the citations can link to them.
PDF_DIR = Path("pdf_store")
PDF_DIR.mkdir(exist_ok=True)
gr.set_static_paths(paths=[str(PDF_DIR)])

EXAMPLE_QUESTIONS = [
    "How many attention heads does the base model use?",
    "What BLEU score did the model get on English to German translation?",
]


def citation(source, page):
    """A clickable link to the PDF page. Plain text if the PDF copy is missing."""
    if (PDF_DIR / source).exists():
        link = f"/gradio_api/file={PDF_DIR.as_posix()}/{quote(source)}#page={page}"
        return f"From: [{source}, page {page}]({link})"
    return f"From: {source}, page {page}"


def upload_pdfs(files):
    if not files:
        return "No file chosen. Pick one or more PDFs first."

    lines = []
    for path in files:
        name = Path(path).name
        try:
            if Path(path).resolve() != (PDF_DIR / name).resolve():
                shutil.copy(path, PDF_DIR / name)
            count = index_pdf(path)
            lines.append(f"Indexed {name}: {count} chunks")
        except Exception as error:
            lines.append(f"Could not read {name}: {error}")
    lines.append(f"Total chunks in database: {collection.count()}")
    return "\n".join(lines)


def chat(question, history):
    history = history or []
    if not question.strip():
        return "", history

    try:
        text, sources = answer(question, history)
    except Exception as error:
        text, sources = f"Something went wrong: {error}", []

    if sources:
        text += "\n\n" + "\n\n".join(citation(source, page) for source, page in sources)

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": text},
    ]
    return "", history


with gr.Blocks(title="DocuChat") as demo:
    gr.Markdown(
        "# DocuChat\n"
        "Ask questions about your PDFs. Everything runs on your laptop.\n\n"
        "1. Upload one or more PDFs and click **Index PDFs**.\n"
        "2. Type a question and click **Ask**.\n"
        "3. Each answer shows the file and page it came from."
    )

    files = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
    index_button = gr.Button("Index PDFs")
    status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot(label="Chat")
    box = gr.Textbox(label="Your question", placeholder="Type your question here")
    ask_button = gr.Button("Ask")
    clear_button = gr.Button("Clear History")
    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=box)

    index_button.click(upload_pdfs, files, status)
    ask_button.click(chat, [box, chatbot], [box, chatbot])
    box.submit(chat, [box, chatbot], [box, chatbot])
    clear_button.click(lambda: ([], ""), None, [chatbot, box])


if __name__ == "__main__":
    demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))