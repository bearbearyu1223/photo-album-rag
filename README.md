# Photo Album RAG

A fully local, open-source RAG system for your photo library, based on
Anthropic's [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) technique.

## How it works

```
INDEXING (run once)
────────────────────────────────────────────────────────
Photo
  → EXIF (date, GPS, camera)
  → GPS → human-readable location  (geopy + Nominatim)
  → Vision caption                 (LLaVA via Ollama)
  → Contextual description         (EXIF + caption merged)
  → Embedding                      (sentence-transformers)
  → ChromaDB vector store  +  BM25 index

QUERYING (run any time)
────────────────────────────────────────────────────────
Natural language query
  → Stage 1a: Semantic search      (ChromaDB, top 50)
  → Stage 1b: BM25 lexical search  (rank_bm25, top 50)
  → Stage 2:  Reciprocal Rank Fusion → merged top 100
  → Stage 3:  Cross-encoder rerank → top 20 → top K
  → Stage 4:  Local LLM answers    (Llama 3.2 via Ollama)
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

```bash
# Install from https://ollama.com, then:
ollama pull llava        # vision model for captioning (~4 GB)
ollama pull llama3.2     # LLM for answering queries
```

`moondream` is a lighter alternative to `llava` for faster indexing:
```bash
ollama pull moondream
# then set VISION_MODEL = "moondream" in src/photo_rag/ingest.py
```

### 3. Clone and set up the project

```bash
git clone https://github.com/bearbearyu1223/photo-album-rag.git
cd photo-album-rag

# Create venv and install all dependencies (reads pyproject.toml)
uv sync

# Activate the venv
source .venv/bin/activate
```

### 4. Optional extras

```bash
uv sync --extra clip   # CLIP image-similarity search
uv sync --extra ui     # Streamlit web UI
uv sync --extra all    # Everything
uv sync --dev          # Dev tools (pytest, ruff, mypy)
```

## Usage

### Index your photos

```bash
# With activated venv:
photo-ingest --photos ~/Pictures --index ./photo_index

# Or without activating the venv:
uv run photo-ingest --photos ~/Pictures --index ./photo_index
```

This is the slow step (~10–30 seconds per photo). The index is **incremental** — re-running skips already-indexed photos.

### Query — interactive REPL

```bash
photo-query --index ./photo_index
```

### Query — single question

```bash
photo-query --index ./photo_index --query "beach photos from last summer"
```

### Retrieval only (no LLM answer)

```bash
photo-search --index ./photo_index --query "hiking with friends" --top-k 5
```

## Example queries

- "Show me photos from Christmas 2023"
- "Pictures taken in Japan"
- "Photos with dogs or pets"
- "Candid shots at parties or celebrations"
- "Outdoor hiking or nature photos"
- "Any group photos with lots of people?"

## Project structure

```
photo-album-rag/
├── pyproject.toml          # project metadata + dependencies (uv)
├── README.md
├── src/
│   └── photo_rag/
│       ├── __init__.py
│       ├── ingest.py       # indexing pipeline
│       ├── retrieve.py     # hybrid retrieval (semantic + BM25 + rerank)
│       └── query.py        # LLM-powered query interface
└── tests/                  # add your tests here
```

## Development

```bash
uv sync --dev

# Lint + format
uv run ruff check src/
uv run ruff format src/

# Type check
uv run mypy src/

# Run tests
uv run pytest
```

## Tips

- **Caption quality matters most.** Edit `CAPTION_PROMPT` in `ingest.py` to match your library.
- **HEIC support** is included via `pillow-heif` — no extra steps for iPhone photos.
- The `photo_index/` directory is excluded from git via `.gitignore`.
- For better reranking accuracy, swap `cross-encoder/ms-marco-MiniLM-L-6-v2` for `cross-encoder/ms-marco-electra-base` in `retrieve.py`.
