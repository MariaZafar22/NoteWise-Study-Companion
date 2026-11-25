import csv
import io
import tempfile
from pathlib import Path

import streamlit as st

from src import config, study_tools
from src.loader import SUPPORTED, load_document
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="NoteWise Study Companion", page_icon="🎓", layout="wide")
st.markdown(
    "<style>h1,h2,h3{font-weight:800!important}.stChatMessage{border-radius:14px}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    return RAGPipeline()


pipeline = get_pipeline()


def show_sources(sources):
    with st.expander("Sources from your notes"):
        for s in sources:
            st.markdown(f"**{s['source']}** · page {s['page']} · score {s['score']}")
            st.caption(s["text"][:350] + "...")


def run_llm_feature(label, func, *args):
    try:
        with st.spinner(f"Creating your {label}..."):
            return func(*args)
    except Exception as error:
        st.error(f"Could not create the {label}. Check your Gemini key and connection. ({type(error).__name__})")
        return None


st.title("🎓 NoteWise Study Companion")
st.caption("Your notes → Chunks → Embeddings → Vector DB → Retrieval → Gemini → Answers, quizzes & flashcards")

with st.sidebar:
    st.header("📄 Your notes")
    files = st.file_uploader(
        "Upload PDF, TXT or MD", type=[e.strip(".") for e in SUPPORTED], accept_multiple_files=True
    )
    chunk_size = st.slider("Chunk size", 300, 1500, config.CHUNK_SIZE, 50)
    overlap = st.slider("Chunk overlap", 0, 400, config.CHUNK_OVERLAP, 10)
    top_k = st.slider("Passages to retrieve", 1, 8, config.TOP_K)
    if st.button("Build study index", type="primary", use_container_width=True) and files:
        total = 0
        with st.spinner("Reading, chunking and embedding..."):
            for f in files:
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / f.name
                    path.write_bytes(f.getvalue())
                    total += pipeline.ingest_pages(load_document(path, f.name), chunk_size, overlap)
        st.success(f"Indexed {total} chunks.")
    st.metric("Chunks stored", pipeline.store.count())
    for name in pipeline.store.sources():
        st.write(f"• {name}")
    if st.button("Clear all notes", use_container_width=True):
        pipeline.store.reset()
        for key in ("history", "quiz", "cards", "summary"):
            st.session_state.pop(key, None)
        st.rerun()

if pipeline.store.count() == 0:
    st.info("Upload your lecture notes in the sidebar and press **Build study index**. "
            "You can also try the files in the `sample_docs` folder.")

tab_ask, tab_quiz, tab_cards, tab_sum = st.tabs(["💬 Ask", "📝 Quiz", "🗂️ Flashcards", "📌 Summary"])

# ---------------- Ask ----------------
with tab_ask:
    history = st.session_state.setdefault("history", [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                show_sources(turn["sources"])
    if question := st.chat_input("Ask anything from your notes"):
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching your notes..."):
                result = pipeline.ask(question, top_k)
            st.markdown(result["answer"])
            if result["sources"]:
                show_sources(result["sources"])
        history.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})

# ---------------- Quiz ----------------
with tab_quiz:
    topic = st.text_input("Quiz topic", placeholder="e.g. deadlocks, overfitting", key="quiz_topic")
    count = st.slider("Number of questions", 3, 10, 5, key="quiz_n")
    if st.button("Generate quiz") and topic:
        quiz = run_llm_feature("quiz", study_tools.generate_quiz, pipeline, topic, count)
        if quiz is not None:
            st.session_state["quiz"] = quiz
            if not quiz:
                st.warning("I could not find enough about that topic in your notes.")
    quiz = st.session_state.get("quiz", [])
    if quiz:
        with st.form("quiz_form"):
            for i, q in enumerate(quiz):
                st.markdown(f"**Q{i + 1}. {q['question']}**")
                st.radio("Choose one", q["options"], index=None, key=f"answer_{i}", label_visibility="collapsed")
            submitted = st.form_submit_button("Check my answers")
        if submitted:
            score = 0
            for i, q in enumerate(quiz):
                chosen = st.session_state.get(f"answer_{i}")
                correct = q["options"][q["answer_index"]]
                if chosen == correct:
                    score += 1
                    st.success(f"Q{i + 1}: Correct ✅  {q['explanation']}")
                else:
                    st.error(f"Q{i + 1}: Correct answer is **{correct}**. {q['explanation']}")
            st.subheader(f"Score: {score} / {len(quiz)}")

# ---------------- Flashcards ----------------
with tab_cards:
    topic = st.text_input("Flashcard topic", placeholder="e.g. paging, supervised learning", key="cards_topic")
    count = st.slider("Number of cards", 4, 15, 8, key="cards_n")
    if st.button("Generate flashcards") and topic:
        cards = run_llm_feature("flashcards", study_tools.generate_flashcards, pipeline, topic, count)
        if cards is not None:
            st.session_state["cards"] = cards
            if not cards:
                st.warning("I could not find enough about that topic in your notes.")
    cards = st.session_state.get("cards", [])
    for i, card in enumerate(cards, 1):
        with st.expander(f"Card {i}: {card['front']}"):
            st.markdown(card["back"])
    if cards:
        buffer = io.StringIO()
        csv.writer(buffer).writerows([[c["front"], c["back"]] for c in cards])
        st.download_button("⬇️ Download as CSV (Anki ready)", buffer.getvalue(), "flashcards.csv", "text/csv")

# ---------------- Summary ----------------
with tab_sum:
    topic = st.text_input("Summary topic", placeholder="e.g. process scheduling", key="sum_topic")
    if st.button("Summarize topic") and topic:
        summary = run_llm_feature("summary", study_tools.summarize_topic, pipeline, topic)
        if summary is not None:
            st.session_state["summary"] = summary or "I could not find that topic in your notes."
    if st.session_state.get("summary"):
        st.markdown(st.session_state["summary"])
