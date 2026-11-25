"""Study features built on top of retrieval: quiz, flashcards and topic summary."""
import json

from . import llm

QUIZ_PROMPT = """You are a study-quiz writer. Using ONLY the notes below, write {n} multiple-choice
questions about "{topic}". Return ONLY a JSON array, with no markdown and no extra text.
Each item must look like:
{{"question": "...", "options": ["A text", "B text", "C text", "D text"], "answer_index": 0, "explanation": "..."}}
answer_index is the position (0-3) of the correct option. Make wrong options believable.

NOTES:
{context}
"""

CARDS_PROMPT = """You are a flashcard writer. Using ONLY the notes below, write {n} flashcards about
"{topic}". Return ONLY a JSON array, with no markdown and no extra text.
Each item must look like: {{"front": "short question or term", "back": "short, clear answer"}}

NOTES:
{context}
"""

SUMMARY_PROMPT = """You are a study assistant. Using ONLY the notes below, write a revision summary of
"{topic}" as 5 to 8 short bullet points, easy to memorise. Start each line with "- ".

NOTES:
{context}
"""


def parse_json_list(text):
    """Pull a JSON array out of a model reply, even if it is wrapped in code fences."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def clean_quiz(items):
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        options = item.get("options")
        index = item.get("answer_index")
        if (
            isinstance(item.get("question"), str)
            and isinstance(options, list)
            and len(options) == 4
            and isinstance(index, int)
            and 0 <= index <= 3
        ):
            valid.append({
                "question": item["question"].strip(),
                "options": [str(o).strip() for o in options],
                "answer_index": index,
                "explanation": str(item.get("explanation", "")).strip(),
            })
    return valid


def clean_cards(items):
    return [
        {"front": str(i["front"]).strip(), "back": str(i["back"]).strip()}
        for i in items
        if isinstance(i, dict) and i.get("front") and i.get("back")
    ]


def _context_for(pipeline, topic, top_k=8):
    hits = pipeline.retrieve(topic, top_k=top_k)
    return hits, llm.format_context(hits)


def generate_quiz(pipeline, topic, n=5):
    hits, context = _context_for(pipeline, topic)
    if not hits:
        return []
    reply = llm.generate_text(QUIZ_PROMPT.format(n=n, topic=topic, context=context))
    return clean_quiz(parse_json_list(reply))[:n]


def generate_flashcards(pipeline, topic, n=8):
    hits, context = _context_for(pipeline, topic)
    if not hits:
        return []
    reply = llm.generate_text(CARDS_PROMPT.format(n=n, topic=topic, context=context))
    return clean_cards(parse_json_list(reply))[:n]


def summarize_topic(pipeline, topic):
    hits, context = _context_for(pipeline, topic)
    if not hits:
        return ""
    return llm.generate_text(SUMMARY_PROMPT.format(topic=topic, context=context))
