"""
retrieve.py — Hybrid Retrieval with Contextual Embeddings + BM25 + Reranking
==============================================================================
Implements the three-stage retrieval pipeline from Anthropic's Contextual
Retrieval blog post, adapted for photos:

  Stage 1 — Broad recall:
      Semantic search (ChromaDB cosine similarity) + BM25 lexical search,
      each returning top-N candidates.

  Stage 2 — Rank fusion:
      Reciprocal Rank Fusion (RRF) merges both result lists into a single
      unified ranking without needing to tune score scales.

  Stage 3 — Precision reranking:
      A cross-encoder reranker scores each candidate against the query
      and selects the final top-K to return.

Usage (as a library):
    from retrieve import PhotoRetriever
    retriever = PhotoRetriever(index_dir="./photo_index")
    results = retriever.search("beach photos from last summer", top_k=5)
    for r in results:
        print(r["path"], r["score"])

Usage (CLI):
    python retrieve.py --index ./photo_index --query "hiking with friends"
"""

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class PhotoResult:
    photo_id: str
    path: str
    description: str          # full contextual description used for indexing
    caption: str              # just the vision model caption
    datetime: str
    camera: str
    location: str = ""        # human-readable location from reverse geocoding
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    rrf_score: float = 0.0    # after rank fusion
    rerank_score: float = 0.0 # after cross-encoder

    def display(self):
        print(f"  📷 {Path(self.path).name}")
        print(f"     Date    : {self.datetime or 'unknown'}")
        print(f"     Camera  : {self.camera or 'unknown'}")
        if self.location:
            print(f"     Location: {self.location}")
        print(f"     Caption : {self.caption[:120]}...")
        print(f"     Score   : {self.rerank_score:.4f}")
        print()


# ── Retriever ─────────────────────────────────────────────────────────────────

class PhotoRetriever:
    """
    Loads a pre-built ChromaDB + BM25 index and exposes a .search() method
    that returns reranked PhotoResult objects.
    """

    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        index_dir: str | Path,
        semantic_candidates: int = 50,
        bm25_candidates: int = 50,
        rerank_candidates: int = 20,
        clip_candidates: int = 50,
    ):
        """
        Args:
            index_dir:           Directory produced by ingest.py.
            semantic_candidates: How many results to pull from ChromaDB.
            bm25_candidates:     How many results to pull from BM25.
            rerank_candidates:   Pool size passed to the cross-encoder;
                                 search() returns top_k from this pool.
            clip_candidates:     How many results to pull from CLIP index.
                                 Ignored if CLIP was not built during ingest.
        """
        self.index_dir = Path(index_dir)
        self.semantic_candidates = semantic_candidates
        self.bm25_candidates = bm25_candidates
        self.rerank_candidates = rerank_candidates
        self.clip_candidates = clip_candidates

        self._load_index()

    # ── Index loading ──────────────────────────────────────────────────────

    def _load_index(self):
        config_path = self.index_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No config.json found in {self.index_dir}. "
                "Did you run ingest.py first?"
            )

        with open(config_path) as f:
            config = json.load(f)

        print(f"Loading embedding model: {config['embedding_model']}")
        self.embedder = SentenceTransformer(config["embedding_model"])

        print(f"Loading reranker: {self.RERANK_MODEL}")
        self.reranker = CrossEncoder(self.RERANK_MODEL)

        print("Loading ChromaDB collection...")
        chroma = PersistentClient(path=str(self.index_dir / "chroma"))
        self.collection = chroma.get_collection("photos")

        print("Loading BM25 index...")
        bm25_path = self.index_dir / "bm25_state.pkl"
        with open(bm25_path, "rb") as f:
            saved = pickle.load(f)
        self.bm25 = saved["bm25"]
        self.bm25_ids = saved["ids"]
        self.id_to_path = saved["id_to_path"]

        # Load CLIP searcher if index was built with --clip
        self.clip_searcher = None
        if config.get("clip_enabled"):
            try:
                from photo_rag.clip_search import CLIPSearcher
                print("Loading CLIP searcher...")
                self.clip_searcher = CLIPSearcher(self.index_dir)
                print(f"  CLIP collection: {self.clip_searcher.collection.count()} photos")
            except ImportError:
                print("  [CLIP] Extra not installed — skipping CLIP search.")

        print(f"Index ready — {len(self.bm25_ids)} photos.\n")

    # ── Search ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[PhotoResult]:
        """
        Full hybrid search pipeline:
          1. Semantic search via ChromaDB
          2. BM25 lexical search
          3. Reciprocal Rank Fusion
          4. Cross-encoder reranking
          5. Return top_k PhotoResult objects
        """
        # 1. Semantic search
        semantic_hits = self._semantic_search(query)

        # 2. BM25 search
        bm25_hits = self._bm25_search(query)

        # 3. CLIP search (if available)
        clip_hits = self._clip_search(query)

        # 4. Merge via RRF (2 or 3 signals depending on CLIP availability)
        rrf_inputs = [semantic_hits, bm25_hits] + ([clip_hits] if clip_hits else [])
        fused = self._reciprocal_rank_fusion(*rrf_inputs)

        # 5. Rerank the top pool
        pool = fused[:self.rerank_candidates]
        reranked = self._rerank(query, pool)

        return reranked[:top_k]

    # ── CLIP search ────────────────────────────────────────────────────────

    def _clip_search(self, query: str) -> list[PhotoResult]:
        """CLIP text→image search. Returns empty list if CLIP is unavailable."""
        if self.clip_searcher is None:
            return []

        hits = self.clip_searcher.text_search(query, n_results=self.clip_candidates)
        if not hits:
            return []

        ids_to_fetch = [pid for pid, _ in hits]
        fetched = self.collection.get(
            ids=ids_to_fetch,
            include=["documents", "metadatas"],
        )
        meta_by_id = {
            pid: (doc, meta)
            for pid, doc, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            )
        }
        results = []
        for photo_id, score in hits:
            if photo_id not in meta_by_id:
                continue
            doc, meta = meta_by_id[photo_id]
            results.append(PhotoResult(
                photo_id=photo_id,
                path=meta["path"],
                description=doc,
                caption=meta.get("caption", ""),
                datetime=meta.get("datetime", ""),
                camera=meta.get("camera", ""),
                location=meta.get("location", ""),
                gps_lat=meta.get("gps_lat", 0.0),
                gps_lon=meta.get("gps_lon", 0.0),
                rrf_score=score,
            ))
        return results

    def image_search(self, photo_path, top_k: int = 5) -> list[PhotoResult]:
        """Find visually similar photos to a given photo file."""
        if self.clip_searcher is None:
            raise RuntimeError(
                "CLIP index not available. Re-run ingest.py with --clip."
            )
        clip_hits = self.clip_searcher.image_search(photo_path, n_results=top_k * 4)
        ids_to_fetch = [pid for pid, _ in clip_hits]
        fetched = self.collection.get(ids=ids_to_fetch, include=["documents", "metadatas"])
        meta_by_id = {
            pid: (doc, meta)
            for pid, doc, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            )
        }
        results = []
        for photo_id, score in clip_hits:
            if photo_id not in meta_by_id:
                continue
            doc, meta = meta_by_id[photo_id]
            results.append(PhotoResult(
                photo_id=photo_id,
                path=meta["path"],
                description=doc,
                caption=meta.get("caption", ""),
                datetime=meta.get("datetime", ""),
                camera=meta.get("camera", ""),
                location=meta.get("location", ""),
                gps_lat=meta.get("gps_lat", 0.0),
                gps_lon=meta.get("gps_lon", 0.0),
                rrf_score=score,
            ))
        return self._rerank("", results)[:top_k]

    # ── Stage 1: Semantic search ───────────────────────────────────────────

    def _semantic_search(self, query: str) -> list[PhotoResult]:
        """Embed the query and find nearest neighbors in ChromaDB."""
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.semantic_candidates,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for photo_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(PhotoResult(
                photo_id=photo_id,
                path=meta["path"],
                description=doc,
                caption=meta.get("caption", ""),
                datetime=meta.get("datetime", ""),
                camera=meta.get("camera", ""),
                location=meta.get("location", ""),
                gps_lat=meta.get("gps_lat", 0.0),
                gps_lon=meta.get("gps_lon", 0.0),
                rrf_score=1.0 - dist,   # cosine distance → similarity
            ))
        return hits

    # ── Stage 2: BM25 lexical search ──────────────────────────────────────

    def _bm25_search(self, query: str) -> list[PhotoResult]:
        """Tokenize the query and score all documents with BM25."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # Pair scores with IDs, sort descending, take top-N
        ranked = sorted(
            zip(scores, self.bm25_ids), key=lambda x: x[0], reverse=True
        )[: self.bm25_candidates]

        # Fetch metadata from ChromaDB for the BM25 hits
        ids_to_fetch = [pid for _, pid in ranked if pid]
        if not ids_to_fetch:
            return []

        fetched = self.collection.get(
            ids=ids_to_fetch,
            include=["documents", "metadatas"],
        )
        meta_by_id = {
            pid: (doc, meta)
            for pid, doc, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            )
        }

        hits = []
        for bm25_score, photo_id in ranked:
            if photo_id not in meta_by_id:
                continue
            doc, meta = meta_by_id[photo_id]
            hits.append(PhotoResult(
                photo_id=photo_id,
                path=meta["path"],
                description=doc,
                caption=meta.get("caption", ""),
                datetime=meta.get("datetime", ""),
                camera=meta.get("camera", ""),
                location=meta.get("location", ""),
                gps_lat=meta.get("gps_lat", 0.0),
                gps_lon=meta.get("gps_lon", 0.0),
                rrf_score=bm25_score,
            ))
        return hits

    # ── Stage 3: Reciprocal Rank Fusion ───────────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(
        *result_lists: list[PhotoResult],
        k: int = 60,
    ) -> list[PhotoResult]:
        """
        Merge multiple ranked lists into one using RRF.
        RRF score = Σ  1 / (k + rank_i)

        k=60 is the standard default; higher k reduces the impact of top ranks.
        The beauty of RRF is that it's scale-free — you can safely fuse a
        cosine similarity list and a BM25 score list without normalizing.
        """
        rrf_scores: dict[str, float] = {}
        photo_by_id: dict[str, PhotoResult] = {}

        for result_list in result_lists:
            for rank, result in enumerate(result_list, start=1):
                pid = result.photo_id
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)
                photo_by_id[pid] = result

        # Assign fused scores and sort
        fused = []
        for pid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            photo = photo_by_id[pid]
            photo.rrf_score = score
            fused.append(photo)

        return fused

    # ── Stage 4: Cross-encoder reranking ──────────────────────────────────

    def _rerank(self, query: str, candidates: list[PhotoResult]) -> list[PhotoResult]:
        """
        Score each (query, description) pair with a cross-encoder.
        Cross-encoders are slower than bi-encoders but much more accurate
        because they see both texts jointly.
        """
        if not candidates:
            return []

        pairs = [(query, c.description) for c in candidates]
        scores = self.reranker.predict(pairs)

        for result, score in zip(candidates, scores):
            result.rerank_score = float(score)

        return sorted(candidates, key=lambda r: r.rerank_score, reverse=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Search your indexed photo library.")
    parser.add_argument("--index", type=Path, default=Path("./photo_index"))
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = PhotoRetriever(index_dir=args.index)
    results = retriever.search(args.query, top_k=args.top_k)

    print(f"\nTop {len(results)} results for: \"{args.query}\"\n")
    for result in results:
        result.display()


if __name__ == "__main__":
    main()
