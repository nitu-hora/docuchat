"""Test questions shared by eval.py and experiment.py.

Each question names the file and the PDF page(s) where the answer really is
(page numbers were checked by hand in the PDFs). "any_of" are words that a
correct answer should contain (lowercase).
"""

QUESTIONS = [
    {"question": "How many identical layers does the Transformer encoder have?",
     "source": "attention.pdf", "pages": [3], "any_of": ["6", "six"]},
    {"question": "How many attention heads does the base model use?",
     "source": "attention.pdf", "pages": [5], "any_of": ["8", "eight"]},
    {"question": "What dropout rate does the base model use?",
     "source": "attention.pdf", "pages": [8], "any_of": ["0.1"]},
    {"question": "What BLEU score did the Transformer get on English to German translation?",
     "source": "attention.pdf", "pages": [1, 8], "any_of": ["28.4"]},
    {"question": "Which Wikipedia dump does RAG use for its document index?",
     "source": "rag_paper.pdf", "pages": [4, 7], "any_of": ["december 2018"]},
    {"question": "How many documents are in the RAG Wikipedia index?",
     "source": "rag_paper.pdf", "pages": [4, 19], "any_of": ["21m", "21 million"]},
    {"question": "How are the LoRA matrices A and B initialized?",
     "source": "lora.pdf", "pages": [3], "any_of": ["gaussian"]},
    {"question": "How is the subspace similarity measure in LoRA related to the standard Projection Metric?",
     "source": "lora.pdf", "pages": [17], "any_of": ["projection metric"]},
    {"question": "How was the data collected in the study of students learning languages with mobile devices?",
     "source": "ej1172284.pdf", "pages": [4], "any_of": ["semi-structured", "semi structured"]},
    {"question": "In which countries does Infosys carry out CSR initiatives outside India?",
     "source": "infosys_board_report.pdf", "pages": [11], "any_of": ["australia"]},
    {"question": "What is the capital of France?", "refuse": True},
    {"question": "How many parameters does GPT-4 have?", "refuse": True},
]

# 10 more answerable questions. Used only by experiment.py, so that the
# retrieval experiment runs on 20 answerable questions in total.
EXTRA_QUESTIONS = [
    {"question": "What is the dimension of each attention head in the base model?",
     "source": "attention.pdf", "pages": [5], "any_of": ["64"]},
    {"question": "By how much does LoRA reduce the GPU memory (VRAM) needed to train GPT-3 175B?",
     "source": "lora.pdf", "pages": [4], "any_of": ["350gb", "350 gb"]},
    {"question": "How small can the LoRA trainable parameters be, as a percent of the original GPT-3 parameters?",
     "source": "lora.pdf", "pages": [3], "any_of": ["0.01"]},
    {"question": "What is the difference between RAG-Sequence and RAG-Token?",
     "source": "rag_paper.pdf", "pages": [2, 3], "any_of": ["same document"]},
    {"question": "What score does RAG-Sequence get on Natural Questions (NQ)?",
     "source": "rag_paper.pdf", "pages": [6, 19], "any_of": ["44.5"]},
    {"question": "How many students took part in the interview study?",
     "source": "ej1172284.pdf", "pages": [1, 4], "any_of": ["20", "twenty"]},
    {"question": "How many employees did Infosys have on a consolidated basis at March 31, 2026?",
     "source": "infosys_board_report.pdf", "pages": [8], "any_of": ["3,28,594", "328,594"]},
    {"question": "What was the total dividend per share for fiscal 2026?",
     "source": "infosys_board_report.pdf", "pages": [6], "any_of": ["48"]},
    {"question": "How many committees did the Infosys Board have as on March 31, 2026?",
     "source": "infosys_board_report.pdf", "pages": [10], "any_of": ["six"]},
    {"question": "How much of its free cash flow does Infosys plan to return to shareholders over five years?",
     "source": "infosys_board_report.pdf", "pages": [4, 6], "any_of": ["85%"]},
]
