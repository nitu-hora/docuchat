"""Offline tests: they need no Ollama and no internet (tiktoken must already be downloaded).

Run from the project folder:   python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["DB_PATH"] = tempfile.mkdtemp()          # never touch the real chroma_db

import chromadb  # noqa: E402

import ask  # noqa: E402
import parse_pdf  # noqa: E402
from questions import EXTRA_QUESTIONS, QUESTIONS  # noqa: E402
from retrieval import Retriever, fuse, tokenize  # noqa: E402


def fake_vector(text):
    """Tiny stand-in for an embedding: counts a few letters. Same text, same vector."""
    return [float(text.lower().count(letter)) for letter in "abcdefghij"] or [0.0]


def make_collection(rows):
    client = chromadb.PersistentClient(path=tempfile.mkdtemp())
    collection = client.get_or_create_collection("test_chunks")
    for number, (text, page) in enumerate(rows):
        collection.add(ids=[f"c{number}"], documents=[text], embeddings=[fake_vector(text)],
                       metadatas=[{"source": "paper.pdf", "page": page}])
    return collection


class TokenizerTests(unittest.TestCase):
    def test_ligature_becomes_plain_letters(self):
        self.assertEqual(tokenize("ﬁne"), ["fine"])

    def test_numbers_stay_whole_and_stop_words_go(self):
        words = tokenize("What is the BLEU score 28.4 and 3,28,594?")
        self.assertIn("28.4", words)
        self.assertIn("3,28,594", words)
        self.assertNotIn("the", words)


class MergeTests(unittest.TestCase):
    def test_chunk_found_by_both_searches_wins(self):
        def chunk(name):
            return {"id": name, "text": "", "source": "s", "page": 1, "distance": None}
        merged = fuse([[chunk("x"), chunk("y")], [chunk("z"), chunk("y")]])
        self.assertEqual(merged[0]["id"], "y")


class RetrieverTests(unittest.TestCase):
    def test_empty_database_gives_nothing(self):
        client = chromadb.PersistentClient(path=tempfile.mkdtemp())
        retriever = Retriever(client.get_or_create_collection("empty_one"), fake_vector,
                              hybrid=False, rerank=False, rewrite=False)
        self.assertEqual(retriever.retrieve("anything"), ([], None))

    def test_asking_for_more_chunks_than_exist_is_fine(self):
        retriever = Retriever(make_collection([("alpha", 1), ("beta", 2)]), fake_vector,
                              hybrid=False, rerank=False, rewrite=False)
        chunks, best = retriever.retrieve("alpha", top_k=50)
        self.assertEqual(len(chunks), 2)
        self.assertIsNotNone(best)

    def test_hybrid_finds_an_exact_word_that_vector_search_misses(self):
        rows = [("the retriever uses a bi-encoder", 1), ("BART is the generator", 2),
                ("training details and data", 3), ("results table", 4)]
        collection = make_collection(rows)
        same_vector = lambda text: [1.0] * 10          # vector search cannot tell chunks apart
        hybrid = Retriever(collection, same_vector, hybrid=True, rerank=False, rewrite=False)
        chunks, _ = hybrid.retrieve("Which part is BART?", top_k=1)
        self.assertEqual(chunks[0]["page"], 2)

    def test_keyword_index_sees_newly_added_chunks(self):
        collection = make_collection([("first chunk", 1)])
        retriever = Retriever(collection, fake_vector, hybrid=True, rerank=False, rewrite=False)
        retriever.retrieve("first")
        collection.add(ids=["new"], documents=["zebra stripes"], embeddings=[fake_vector("zebra stripes")],
                       metadatas=[{"source": "paper.pdf", "page": 9}])
        chunks, _ = retriever.retrieve("zebra", top_k=3)
        self.assertIn(9, [chunk["page"] for chunk in chunks])


class CitationTests(unittest.TestCase):
    def test_copied_phrase_is_detected(self):
        chunk = "In this work we employ h = 8 parallel attention layers, or heads."
        self.assertTrue(ask.shares_phrase("We employ h = 8 parallel attention layers.", chunk))
        self.assertFalse(ask.shares_phrase("Eight heads are used by the model.", chunk))

    def _run(self, reply_text, best=0.4):
        chunks = [
            {"id": "a", "text": "Some other text about training.", "source": "p.pdf", "page": 3, "distance": best},
            {"id": "b", "text": "We use h = 8 parallel attention layers in this model.", "source": "p.pdf", "page": 5, "distance": best},
        ]
        ask.retriever.retrieve = lambda question, top_k, history=None: (chunks, best)
        reply = type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": reply_text})()})()]})()
        ask.llm.chat.completions.create = lambda **kwargs: reply
        return ask.answer("How many heads?")

    def test_cites_the_chunk_the_answer_copied(self):
        text, cited = self._run("It says h = 8 parallel attention layers in this model.")
        self.assertEqual(cited, [("p.pdf", 5)])

    def test_falls_back_to_closest_chunk(self):
        _, cited = self._run("Eight.")
        self.assertEqual(cited, [("p.pdf", 3)])

    def test_dont_know_shows_no_sources(self):
        _, cited = self._run("I don't know.")
        self.assertEqual(cited, [])

    def test_far_question_is_refused_without_asking_the_model(self):
        def boom(**kwargs):
            raise AssertionError("the model must not be called")
        ask.llm.chat.completions.create = boom
        ask.retriever.retrieve = lambda question, top_k, history=None: (
            [{"id": "a", "text": "t", "source": "p.pdf", "page": 1, "distance": 1.0}], 1.0)
        text, cited = ask.answer("What is the capital of France?")
        self.assertEqual(text, "I could not find this in your documents.")
        self.assertEqual(cited, [])


class ChunkingTests(unittest.TestCase):
    def test_no_empty_chunks_and_no_lost_sentences(self):
        text = " ".join(f"Sentence number {i} says something useful." for i in range(200))
        chunks = parse_pdf.split_into_chunks(text)
        self.assertTrue(all(chunk.strip() for chunk in chunks))
        joined = " ".join(chunks)
        for i in range(200):
            self.assertIn(f"Sentence number {i} says", joined)

    def test_chunks_stay_near_the_size_limit(self):
        text = " ".join(f"Sentence number {i} says something useful." for i in range(400))
        for chunk in parse_pdf.split_into_chunks(text):
            self.assertLessEqual(parse_pdf.count_tokens(chunk), parse_pdf.CHUNK_TOKENS + 20)

    def test_consecutive_chunks_overlap(self):
        text = " ".join(f"Sentence number {i} says something useful." for i in range(400))
        chunks = parse_pdf.split_into_chunks(text)
        self.assertGreater(len(chunks), 1)
        last_sentence_of_first = chunks[0].split(". ")[-1].strip()
        self.assertIn(last_sentence_of_first[:20], chunks[1])


class QuestionFileTests(unittest.TestCase):
    def test_every_question_is_complete_and_unique(self):
        everything = QUESTIONS + EXTRA_QUESTIONS
        self.assertEqual(len({q["question"] for q in everything}), len(everything))
        for item in everything:
            if item.get("refuse"):
                continue
            self.assertTrue(item["source"].endswith(".pdf"))
            self.assertTrue(item["pages"] and all(isinstance(p, int) and p > 0 for p in item["pages"]))
            self.assertTrue(item["any_of"])


if __name__ == "__main__":
    unittest.main()
