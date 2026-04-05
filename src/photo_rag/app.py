"""
app.py — Streamlit Web UI for Photo Album RAG
==============================================
Provides a browser-based interface for searching your photo library.
Features:
  - Natural language text search
  - "Find similar photos" (visual similarity via CLIP, if indexed)
  - Thumbnail grid with expandable metadata
  - LLM-generated answer for text queries
  - Sidebar controls for index path, top-k, and search mode

Run:
    uv run streamlit run src/photo_rag/app.py
    streamlit run src/photo_rag/app.py -- --index ./photo_index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Photo Album RAG",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Tighten up the card grid */
    .photo-grid { display: flex; flex-wrap: wrap; gap: 12px; }

    /* Score badge */
    .score-badge {
        display: inline-block;
        background: #1a1a2e;
        color: #e0e0ff;
        font-size: 0.72rem;
        font-family: monospace;
        padding: 2px 7px;
        border-radius: 4px;
        margin-bottom: 4px;
    }

    /* Pill for signal tags */
    .signal-pill {
        display: inline-block;
        font-size: 0.65rem;
        padding: 1px 6px;
        border-radius: 10px;
        margin-right: 3px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .pill-semantic { background: #dbeafe; color: #1e40af; }
    .pill-bm25     { background: #dcfce7; color: #166534; }
    .pill-clip     { background: #fef3c7; color: #92400e; }

    /* Minimal meta table */
    .meta-row { font-size: 0.78rem; color: #6b7280; margin-bottom: 2px; }

    /* Answer box */
    .answer-box {
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ── Retriever caching ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading index…")
def load_retriever(index_dir: str):
    """Load once and cache across Streamlit reruns."""
    from photo_rag.retrieve import PhotoRetriever
    return PhotoRetriever(index_dir=index_dir)


# ── LLM answer ───────────────────────────────────────────────────────────────

def get_llm_answer(query: str, results) -> str:
    """Call local Ollama LLM to synthesize an answer from retrieved photos."""
    try:
        import ollama
        from photo_rag.query import answer_query
        return answer_query(query, results)
    except Exception as e:
        return f"*(LLM answer unavailable: {e})*"


# ── Thumbnail helper ──────────────────────────────────────────────────────────

def load_thumbnail(path: str, size: int = 300) -> Image.Image | None:
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img.thumbnail((size, size), Image.LANCZOS)
        return img
    except Exception:
        return None


# ── Result card renderer ──────────────────────────────────────────────────────

def render_result_card(result, col, rank: int):
    """Render one photo result into a Streamlit column."""
    with col:
        # Thumbnail
        thumb = load_thumbnail(result.path)
        if thumb:
            st.image(thumb, use_container_width=True)
        else:
            st.markdown(
                f"<div style='height:180px;background:#e5e7eb;display:flex;"
                f"align-items:center;justify-content:center;border-radius:6px;"
                f"color:#9ca3af;font-size:0.8rem;'>No preview</div>",
                unsafe_allow_html=True,
            )

        # Filename + rank
        filename = Path(result.path).name
        st.markdown(f"**{rank}. {filename}**")

        # Score badge
        st.markdown(
            f'<span class="score-badge">score {result.rerank_score:.3f}</span>',
            unsafe_allow_html=True,
        )

        # Metadata rows
        if result.datetime:
            st.markdown(
                f'<div class="meta-row">📅 {result.datetime[:10]}</div>',
                unsafe_allow_html=True,
            )
        if result.camera:
            st.markdown(
                f'<div class="meta-row">📷 {result.camera}</div>',
                unsafe_allow_html=True,
            )
        if result.location:
            st.markdown(
                f'<div class="meta-row">📍 {result.location}</div>',
                unsafe_allow_html=True,
            )
        elif result.gps_lat and result.gps_lon:
            st.markdown(
                f'<div class="meta-row">📍 {result.gps_lat:.4f}, {result.gps_lon:.4f}</div>',
                unsafe_allow_html=True,
            )

        # Caption in expander
        if result.caption:
            with st.expander("Caption", expanded=False):
                st.write(result.caption)

        # "Find similar" button (CLIP)
        if st.button("🔍 Find similar", key=f"sim_{result.photo_id}", use_container_width=True):
            st.session_state["similar_to"] = result.path
            st.session_state["search_mode"] = "similar"
            st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, int, bool]:
    with st.sidebar:
        st.title("📷 Photo RAG")
        st.caption("Contextual retrieval for your photo library")
        st.divider()

        index_dir = st.text_input(
            "Index directory",
            value=str(Path("./photo_index").resolve()),
            help="Path to the index created by photo-ingest",
        )

        top_k = st.slider("Results to show", min_value=1, max_value=20, value=6)

        show_answer = st.toggle("Show LLM answer", value=True,
                                help="Generate a natural-language answer using Ollama")

        st.divider()
        st.markdown("**How to index your photos:**")
        st.code("photo-ingest --photos ~/Pictures\\\n  --index ./photo_index", language="bash")
        st.markdown("**With CLIP support:**")
        st.code("photo-ingest --photos ~/Pictures\\\n  --index ./photo_index --clip", language="bash")
        st.divider()
        st.caption("Runs fully locally — no data leaves your machine.")

    return index_dir, top_k, show_answer


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    index_dir, top_k, show_answer = render_sidebar()

    # ── Load retriever ────────────────────────────────────────────────────────
    try:
        retriever = load_retriever(index_dir)
    except FileNotFoundError:
        st.error(
            f"No index found at **{index_dir}**. "
            "Run `photo-ingest --photos ~/Pictures --index ./photo_index` first."
        )
        st.stop()

    clip_available = getattr(retriever, "clip_searcher", None) is not None

    # ── Search mode tabs ──────────────────────────────────────────────────────
    tab_text, tab_similar = st.tabs(["🔎 Text search", "🖼️ Similar photos"])

    # ═══════════════════════════════════════════════════
    # Tab 1 — Text search
    # ═══════════════════════════════════════════════════
    with tab_text:
        st.markdown("### Search your photo library")

        query = st.text_input(
            "What are you looking for?",
            placeholder="e.g. beach photos from last summer",
            label_visibility="collapsed",
        )

        # Example queries
        st.markdown("**Try:**")
        example_cols = st.columns(4)
        examples = [
            "Christmas with family",
            "hiking or outdoor adventure",
            "photos taken in Japan",
            "birthday celebrations",
        ]
        for i, (col, ex) in enumerate(zip(example_cols, examples)):
            with col:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    query = ex

        if query:
            with st.spinner(f'Searching for "{query}"…'):
                results = retriever.search(query, top_k=top_k)

            if not results:
                st.info("No photos found. Try different search terms.")
            else:
                # LLM answer
                if show_answer:
                    with st.spinner("Generating answer…"):
                        answer = get_llm_answer(query, results)
                    st.markdown(
                        f'<div class="answer-box">💬 {answer}</div>',
                        unsafe_allow_html=True,
                    )

                # Retrieval signal legend
                signal_html = (
                    '<span class="signal-pill pill-semantic">Semantic</span>'
                    '<span class="signal-pill pill-bm25">BM25</span>'
                )
                if clip_available:
                    signal_html += '<span class="signal-pill pill-clip">CLIP</span>'
                st.markdown(
                    f"**{len(results)} photo(s) found** &nbsp;|&nbsp; "
                    f"Active signals: {signal_html}",
                    unsafe_allow_html=True,
                )
                st.divider()

                # Photo grid (3 columns)
                cols_per_row = 3
                for row_start in range(0, len(results), cols_per_row):
                    row_results = results[row_start : row_start + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for result, col, rank in zip(
                        row_results, cols, range(row_start + 1, row_start + cols_per_row + 1)
                    ):
                        render_result_card(result, col, rank)

    # ═══════════════════════════════════════════════════
    # Tab 2 — Visual similarity (CLIP)
    # ═══════════════════════════════════════════════════
    with tab_similar:
        st.markdown("### Find visually similar photos")

        if not clip_available:
            st.info(
                "CLIP index not available. Re-index with `--clip` to enable visual similarity search:\n\n"
                "```bash\nphoto-ingest --photos ~/Pictures --index ./photo_index --clip\n```"
            )
        else:
            # If triggered by "Find similar" button from text results
            prefilled = st.session_state.get("similar_to", "")

            photo_path_input = st.text_input(
                "Path to a photo file",
                value=prefilled,
                placeholder="/Users/you/Pictures/vacation.jpg",
                help="Paste any photo path from your library to find visually similar photos.",
            )

            uploaded = st.file_uploader(
                "Or upload a photo directly",
                type=["jpg", "jpeg", "png", "heic", "webp"],
            )

            if uploaded or photo_path_input:
                search_path: Path | None = None

                if uploaded:
                    # Save upload to a temp file
                    import tempfile
                    suffix = Path(uploaded.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded.read())
                        search_path = Path(tmp.name)
                    # Show the query photo
                    st.image(Image.open(search_path), caption="Query photo", width=300)
                elif photo_path_input:
                    search_path = Path(photo_path_input.strip())
                    if not search_path.exists():
                        st.error(f"File not found: {search_path}")
                        search_path = None
                    else:
                        st.image(load_thumbnail(str(search_path)), caption="Query photo", width=300)

                if search_path:
                    with st.spinner("Finding similar photos…"):
                        try:
                            results = retriever.image_search(search_path, top_k=top_k)
                        except Exception as e:
                            st.error(f"Image search failed: {e}")
                            results = []

                    if not results:
                        st.info("No similar photos found.")
                    else:
                        st.markdown(f"**{len(results)} similar photo(s)**")
                        st.divider()
                        cols_per_row = 3
                        for row_start in range(0, len(results), cols_per_row):
                            row_results = results[row_start : row_start + cols_per_row]
                            cols = st.columns(cols_per_row)
                            for result, col, rank in zip(
                                row_results, cols,
                                range(row_start + 1, row_start + cols_per_row + 1),
                            ):
                                render_result_card(result, col, rank)

    # Clear session state for similar-photo navigation
    if "similar_to" in st.session_state and st.session_state.get("search_mode") != "similar":
        del st.session_state["similar_to"]


def _cli():
    """Entry point for the `photo-app` console script.

    Launches Streamlit as a subprocess so the app works via `uv run photo-app`.
    """
    import subprocess
    from importlib.resources import files

    app_path = str(files("photo_rag").joinpath("app.py"))
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.headless", "true",
            "--",
        ]
        + sys.argv[1:],
    )


if __name__ == "__main__":
    main()
