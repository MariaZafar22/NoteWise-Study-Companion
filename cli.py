"""Terminal access.
  python cli.py ingest sample_docs
  python cli.py ask "What is a deadlock?"
  python cli.py quiz "deadlocks" -n 3
  python cli.py cards "paging"
"""
import argparse

from src import study_tools
from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="NoteWise Study Companion")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="index a file or folder").add_argument("path")
    ask = sub.add_parser("ask", help="ask a question about your notes")
    ask.add_argument("question")
    for name in ("quiz", "cards", "summary"):
        p = sub.add_parser(name, help=f"generate {name} for a topic")
        p.add_argument("topic")
        p.add_argument("-n", type=int, default=5)
    args = parser.parse_args()

    pipeline = RAGPipeline()
    if args.command == "ingest":
        print(f"Indexed {pipeline.ingest_path(args.path)} chunks. Total in DB: {pipeline.store.count()}")
    elif args.command == "ask":
        result = pipeline.ask(args.question)
        print("\nANSWER:\n" + result["answer"])
        print("\nSOURCES:")
        for hit in result["sources"]:
            print(f"- {hit['source']} (page {hit['page']}, score {hit['score']})")
    elif args.command == "quiz":
        for i, q in enumerate(study_tools.generate_quiz(pipeline, args.topic, args.n), 1):
            print(f"\nQ{i}. {q['question']}")
            for letter, option in zip("ABCD", q["options"]):
                print(f"   {letter}) {option}")
            print(f"   Answer: {'ABCD'[q['answer_index']]}  -  {q['explanation']}")
    elif args.command == "cards":
        for card in study_tools.generate_flashcards(pipeline, args.topic, args.n):
            print(f"\nFRONT: {card['front']}\nBACK:  {card['back']}")
    else:
        print(study_tools.summarize_topic(pipeline, args.topic))


if __name__ == "__main__":
    main()
