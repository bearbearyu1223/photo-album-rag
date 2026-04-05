# Photo Album RAG

A fully local, open-source RAG system for your photo library, based on
Anthropic's [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) technique.

Search your photos with natural language — no data leaves your machine.

## How it works

```
INDEXING (run once)
────────────────────────────────────────────────────────────────
Photo
  -> EXIF extraction          (date, GPS, camera)
  -> Reverse geocoding        (GPS -> human-readable location)
  -> Vision caption           (LLaVA via Ollama)
  -> Contextual description   (EXIF + location + caption merged)
  -> Text embedding           (sentence-transformers)
  -> [Optional] CLIP embedding (open-clip, for visual similarity)
  -> ChromaDB vector store + BM25 index

QUERYING (run any time)
────────────────────────────────────────────────────────────────
Natural language query
  -> Stage 1a: Semantic text search       (ChromaDB, top 50)
  -> Stage 1b: BM25 lexical search        (rank_bm25, top 50)
  -> Stage 1c: CLIP text->image search    (optional, top 50)
  -> Stage 2:  Reciprocal Rank Fusion     (merged + deduplicated)
  -> Stage 3:  Cross-encoder reranking    (top 20 -> top K)
  -> Stage 4:  Local LLM answer           (Llama 3.2 via Ollama)
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com) for local model inference

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Ollama and pull models

Download and install from [ollama.com](https://ollama.com), then pull the
models you need:

```bash
# Vision model for photo captioning (pick one)
ollama pull llava          # higher quality (~4 GB)
ollama pull moondream      # faster and lighter (~1.7 GB)

# LLM for answering queries
ollama pull llama3.2
```

To switch vision models, edit `VISION_MODEL` in `src/photo_rag/ingest.py`.

### 3. Clone and install

```bash
git clone https://github.com/bearbearyu1223/photo-album-rag.git
cd photo-album-rag

uv sync                  # core deps only
uv sync --extra ui       # + Streamlit web UI
uv sync --extra clip     # + CLIP image-similarity search
uv sync --extra all      # everything (UI + CLIP)
uv sync --group dev      # + dev tools (pytest, ruff, mypy)
```

## Usage

### Index your photos

```bash
# Text index only
uv run photo-ingest --photos ~/Pictures --index ./photo_index

# Text + CLIP image embeddings (enables visual similarity search)
uv run photo-ingest --photos ~/Pictures --index ./photo_index --clip
```

The index is **incremental** — re-running only processes new photos. To fully
re-index (e.g. after switching vision models), delete `photo_index/` first.

### Web UI (recommended)

```bash
uv run photo-app
```

Opens at http://localhost:8501. Features:
- Natural language search with LLM-generated answers
- Thumbnail grid with EXIF metadata (date, camera, location)
- "Find similar photos" via CLIP visual similarity (if indexed with `--clip`)

### CLI — interactive REPL

```bash
uv run photo-query --index ./photo_index
```

### CLI — single query

```bash
uv run photo-query --index ./photo_index --query "beach photos from last summer"
```

### CLI — retrieval only (no LLM)

```bash
uv run photo-search --index ./photo_index --query "hiking with friends" --top-k 5
```

## Example queries

- "Show me photos from Christmas 2017"
- "Pictures taken in Japan"
- "Photos with dogs or pets"
- "Candid shots at parties or celebrations"
- "Outdoor hiking or nature photos"
- "Any group photos with lots of people?"

## Project structure

```
photo-album-rag/
├── pyproject.toml              # project metadata + dependencies (uv)
├── uv.lock                     # lockfile (committed)
├── README.md
├── src/
│   └── photo_rag/
│       ├── __init__.py
│       ├── ingest.py           # indexing pipeline: EXIF + caption + embed
│       ├── retrieve.py         # hybrid retrieval: semantic + BM25 + CLIP + rerank
│       ├─�� clip_search.py      # CLIP image embedding (optional)
│       ├── query.py            # CLI + LLM-powered query interface
│       └── app.py              # Streamlit web UI
└── tests/
    ├── conftest.py             # shared fixtures
    ├── test_ingest.py          # EXIF helpers, description builder
    └── test_retrieval.py       # RRF, BM25, PhotoRetriever (mocked)
```

## Development

```bash
uv sync --group dev

uv run ruff check src/          # lint
uv run ruff format src/         # format
uv run mypy src/                # type check
uv run pytest                   # run tests
```

## Tips

- **Caption quality matters most.** Edit `CAPTION_PROMPT` in `ingest.py` to
  tune what the vision model focuses on.
- **HEIC support** is built in via `pillow-heif` — iPhone photos work out of
  the box.
- **Apple Silicon** — Ollama uses the GPU automatically. CLIP also uses MPS
  for accelerated inference when available.
- The `photo_index/` directory is git-ignored (large binary data).
- For better reranking quality, swap `cross-encoder/ms-marco-MiniLM-L-6-v2`
  for `cross-encoder/ms-marco-electra-base` in `retrieve.py`.
