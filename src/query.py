"""
query.py — Natural Language Query Interface
============================================
Ties the retriever to a local Ollama LLM so you can ask questions in plain
English and get a synthesized answer with photo references.

Usage (interactive REPL):
    python query.py --index ./photo_index

Usage (single query):
    python query.py --index ./photo_index --query "photos from my trip to Japan"

Example questions this can handle:
    - "Show me photos from Christmas 2023"
    - "Find pictures of hiking or outdoor adventures"
    - "Photos taken in Seattle"
    - "Candid shots at parties or celebrations"
    - "Any photos with dogs?"
"""

import argparse
import textwrap
from pathlib import Path

import ollama

from retrieve import PhotoRetriever, PhotoResult

# ── Configuration ─────────────────────────────────────────────────────────────

ANSWER_MODEL = "llama3.2"     # or "mistral", "phi3", anything you have in ollama
TOP_K = 5                     # how many photos to pass to the LLM

SYSTEM_PROMPT = """You are a helpful photo assistant. The user will ask you about 
their personal photo library. You will be given descriptions of the most relevant 
photos that were retrieved for their query.

Your job is to:
1. Directly answer their question based on the photo descriptions provided.
2. Reference specific photos by their filename and date.
3. If none of the photos match what they asked for, say so honestly.
4. Keep your response concise and friendly.

Do not make up details that aren't in the descriptions. Do not pretend to see 
photos you haven't been given descriptions for."""


# ── Answer generation ─────────────────────────────────────────────────────────

def format_context(results: list[PhotoResult]) -> str:
    """Render retrieved photo results into a text block for the LLM."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[Photo {i}]")
        lines.append(f"  Filename : {Path(r.path).name}")
        lines.append(f"  Date     : {r.datetime or 'unknown'}")
        lines.append(f"  Camera   : {r.camera or 'unknown'}")
        lines.append(f"  Location : (from description)")
        lines.append(f"  Description: {r.description}")
        lines.append("")
    return "\n".join(lines)


def answer_query(query: str, results: list[PhotoResult]) -> str:
    """Pass retrieved photo descriptions to the local LLM and return its answer."""
    if not results:
        return "I couldn't find any photos matching that description in your library."

    context = format_context(results)
    user_message = (
        f"The user asked: \"{query}\"\n\n"
        f"Here are the most relevant photos found:\n\n"
        f"{context}\n"
        f"Please answer the user's question based on these photos."
    )

    response = ollama.chat(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response["message"]["content"].strip()


# ── Display helpers ───────────────────────────────────────────────────────────

def print_results(results: list[PhotoResult]):
    """Print a compact summary of retrieved photos."""
    print(f"\n{'─' * 60}")
    print(f"  Retrieved {len(results)} photo(s):")
    print(f"{'─' * 60}")
    for i, r in enumerate(results, 1):
        name = Path(r.path).name
        date = r.datetime[:10] if r.datetime else "unknown date"
        score = f"{r.rerank_score:.3f}"
        print(f"  {i}. {name}  [{date}]  rerank={score}")
    print(f"{'─' * 60}\n")


def print_answer(answer: str):
    """Print the LLM answer with light wrapping."""
    print("💬 Answer:")
    for line in answer.splitlines():
        print(textwrap.fill(line, width=72, initial_indent="  ", subsequent_indent="  "))
    print()


# ── REPL ──────────────────────────────────────────────────────────────────────

def run_repl(retriever: PhotoRetriever):
    """Simple interactive query loop."""
    print("\n📷 Photo Library RAG — type your question, or 'quit' to exit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\n🔍 Searching...")
        results = retriever.search(query, top_k=TOP_K)
        print_results(results)

        print("🤖 Thinking...")
        answer = answer_query(query, results)
        print_answer(answer)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ask questions about your indexed photo library."
    )
    parser.add_argument("--index", type=Path, default=Path("./photo_index"))
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single query (omit to start interactive REPL)",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    retriever = PhotoRetriever(index_dir=args.index)

    if args.query:
        results = retriever.search(args.query, top_k=args.top_k)
        print_results(results)
        answer = answer_query(args.query, results)
        print_answer(answer)
    else:
        run_repl(retriever)
