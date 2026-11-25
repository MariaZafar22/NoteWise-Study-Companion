"""Connect retrieved context with an LLM (Gemini)."""
from . import config

SYSTEM_RULES = (
    "You are a careful study assistant. Answer ONLY using the notes below. "
    "If the notes do not contain the answer, say you could not find it in the "
    "uploaded notes. Keep the answer clear and short, and mention the source "
    "file names you used."
)


def format_context(hits):
    return "\n\n".join(
        f"[{i}] (source: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )


def build_prompt(question, hits):
    return f"{SYSTEM_RULES}\n\nNOTES:\n{format_context(hits)}\n\nQUESTION: {question}\n\nANSWER:"


def generate_text(prompt):
    key = config.gemini_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    from google import genai

    client = genai.Client(api_key=key)
    response = client.models.generate_content(model=config.GEMINI_MODEL, contents=prompt)
    return (response.text or "").strip()


def generate_answer(question, hits):
    return generate_text(build_prompt(question, hits))
