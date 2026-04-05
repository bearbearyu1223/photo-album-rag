# Photo Album RAG

A fully local, open-source RAG system for your photo library, based on
Anthropic's Contextual Retrieval technique.

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

## Setup

### 1. Install dependencies

```bash
pip install chromadb rank_bm25 sentence-transformers \
            pillow exifread geopy ollama tqdm
```

### 2. Install Ollama and pull models

```bash
# Install Ollama from https://ollama.com
ollama pull llava        # vision model for captions (~4GB)
ollama pull llama3.2     # LLM for answering queries
```

`moondream` is a lighter alternative to `llava` if you want faster indexing:
```bash
ollama pull moondream
# then set VISION_MODEL = "moondream" in ingest.py
```

### 3. Index your photos

```bash
python ingest.py --photos ~/Pictures --index ./photo_index
```

This is the slow step — expect ~10-30 seconds per photo depending on your
machine and which vision model you use. Run it overnight for large libraries.
The index is incremental: re-running skips already-indexed photos.

### 4. Query

Interactive REPL:
```bash
python query.py --index ./photo_index
```

Single query:
```bash
python query.py --index ./photo_index --query "beach photos from last summer"
```

Retrieval only (no LLM answer):
```bash
python retrieve.py --index ./photo_index --query "hiking with friends" --top-k 5
```

## Example queries

- "Show me photos from Christmas 2023"
- "Pictures taken in Japan"
- "Photos with dogs or pets"
- "Candid shots at parties or celebrations"
- "Outdoor hiking or nature photos"
- "Photos from my trip last July"
- "Any group photos with lots of people?"

## File structure

```
photo_rag/
├── ingest.py      # Indexing pipeline
├── retrieve.py    # Hybrid retrieval (semantic + BM25 + rerank)
├── query.py       # LLM-powered query interface
├── README.md
└── photo_index/   # Created by ingest.py
    ├── chroma/    # ChromaDB vector store
    ├── bm25_state.pkl
    └── config.json
```

## Tips

- **Caption quality matters most.** Edit `CAPTION_PROMPT` in `ingest.py` to
  get more useful captions for your library's content.
- **HEIC support** requires `pillow-heif`:
  `pip install pillow-heif` then add `from pillow_heif import register_heif_opener;
  register_heif_opener()` at the top of `ingest.py`.
- The index is incremental — re-running `ingest.py` only processes new photos.
- The `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker is fast and good.
  For better quality try `cross-encoder/ms-marco-electra-base`.
