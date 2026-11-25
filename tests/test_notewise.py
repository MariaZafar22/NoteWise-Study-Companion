import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import llm, study_tools
from src.chunker import split_text
from src.embeddings import Embedder
from src.rag_pipeline import RAGPipeline

DOCS = Path(__file__).resolve().parent.parent / "sample_docs"


def make_pipeline(tmp_path):
    pipe = RAGPipeline(store_path=str(tmp_path / "db"), embedder=Embedder(backend="hash"))
    pipe.ingest_path(DOCS)
    return pipe


def test_chunking_respects_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 3 and all(len(c) <= 360 for c in chunks)


def test_retrieval_finds_right_notes(tmp_path):
    pipe = make_pipeline(tmp_path)
    hits = pipe.retrieve("What are the four conditions of a deadlock?", top_k=2)
    assert hits[0]["source"] == "operating_systems_notes.txt"
    hits = pipe.retrieve("What is overfitting and how to reduce it?", top_k=2)
    assert hits[0]["source"] == "machine_learning_notes.txt"


def test_reingest_does_not_duplicate(tmp_path):
    pipe = make_pipeline(tmp_path)
    before = pipe.store.count()
    pipe.ingest_path(DOCS)
    assert pipe.store.count() == before


def test_irrelevant_question_has_no_context(tmp_path):
    assert make_pipeline(tmp_path).ask("zzzz qqqq xxxx")["mode"] == "no-context"


def test_parse_json_handles_code_fences():
    raw = "Sure!\n```json\n[{\"front\": \"A\", \"back\": \"B\"}]\n```"
    assert study_tools.parse_json_list(raw) == [{"front": "A", "back": "B"}]
    assert study_tools.parse_json_list("not json at all") == []


def test_quiz_validation_drops_bad_items():
    good = {"question": "Q?", "options": ["a", "b", "c", "d"], "answer_index": 2, "explanation": "e"}
    bad = {"question": "Q?", "options": ["a", "b"], "answer_index": 5}
    assert study_tools.clean_quiz([good, bad, "junk"]) == [good]


def test_quiz_and_cards_with_mocked_llm(tmp_path, monkeypatch):
    pipe = make_pipeline(tmp_path)
    quiz = [{"question": "What is a deadlock?", "options": ["a", "b", "c", "d"], "answer_index": 1, "explanation": "x"}]
    monkeypatch.setattr(llm, "generate_text", lambda prompt: json.dumps(quiz))
    assert study_tools.generate_quiz(pipe, "deadlock process resources") == quiz
    cards = [{"front": "Page fault", "back": "Needed page not in memory"}]
    monkeypatch.setattr(llm, "generate_text", lambda prompt: json.dumps(cards))
    assert study_tools.generate_flashcards(pipe, "paging page fault") == cards


def test_no_notes_means_no_quiz(tmp_path):
    empty = RAGPipeline(store_path=str(tmp_path / "empty"), embedder=Embedder(backend="hash"))
    assert study_tools.generate_quiz(empty, "anything") == []
