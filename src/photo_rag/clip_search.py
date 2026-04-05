"""
clip_search.py — CLIP Image-Similarity Search
===============================================
Adds a 4th retrieval signal alongside semantic text search, BM25, and
cross-encoder reranking. CLIP encodes photos and queries into a *shared*
embedding space, so "beach sunset" can match a photo purely by visual
content even if the caption is thin or missing.

How it fits in the pipeline
────────────────────────────
ingest:    clip_index.add(photo)    →  ChromaDB collection "photos_clip"
retrieve:  PhotoRetriever calls clip_search() → gets ranked list
           → fed into RRF alongside semantic + BM25 lists

Requires the [clip] optional extra:
    uv sync --extra clip

Models (set CLIP_MODEL / CLIP_PRETRAINED in this file):
    - "ViT-B-32" / "openai"        ~350 MB  fast, good quality
    - "ViT-L-14" / "openai"        ~900 MB  slower, better quality
    - "ViT-B-16" / "laion2b_s34b_b88k"  community fine-tune, strong on photos
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from PIL import Image

# ── Model config ──────────────────────────────────────────────────────────────

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_COLLECTION = "photos_clip"

# ── Lazy imports (open_clip is optional) ─────────────────────────────────────

def _load_clip():
    try:
        import open_clip
        import torch
    except ImportError as e:
        raise ImportError(
            "CLIP search requires the [clip] extra. "
            "Run: uv sync --extra clip"
        ) from e
    return open_clip, torch


# ── Indexing ──────────────────────────────────────────────────────────────────

class CLIPIndexer:
    """
    Encodes photos with CLIP and stores image embeddings in a dedicated
    ChromaDB collection alongside the main text collection.
    """

    def __init__(self, index_dir: Path):
        open_clip, torch = _load_clip()
        from chromadb import PersistentClient

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"CLIP running on: {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.torch = torch

        chroma = PersistentClient(path=str(index_dir / "chroma"))
        self.collection = chroma.get_or_create_collection(
            name=CLIP_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, photo_id: str, photo_path: Path):
        """Encode one photo and upsert into the CLIP collection."""
        existing = self.collection.get(ids=[photo_id])["ids"]
        if existing:
            return  # already indexed

        try:
            img = self.preprocess(Image.open(photo_path).convert("RGB"))
            img_tensor = img.unsqueeze(0).to(self.device)

            with self.torch.no_grad():
                embedding = self.model.encode_image(img_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            self.collection.add(
                ids=[photo_id],
                embeddings=[embedding.squeeze().cpu().tolist()],
                metadatas=[{"path": str(photo_path)}],
            )
        except Exception as e:
            print(f"  [CLIP] Could not index {photo_path.name}: {e}")


# ── Retrieval ─────────────────────────────────────────────────────────────────

class CLIPSearcher:
    """
    Loads the CLIP collection and supports:
      - text_search(query)   → find photos semantically matching a text query
      - image_search(path)   → find photos visually similar to a given photo
    """

    def __init__(self, index_dir: Path):
        open_clip, torch = _load_clip()
        from chromadb import PersistentClient

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.torch = torch

        chroma = PersistentClient(path=str(index_dir / "chroma"))
        self.collection = chroma.get_collection(CLIP_COLLECTION)

    def text_search(
        self, query: str, n_results: int = 50
    ) -> list[tuple[str, float]]:
        """
        Encode the text query with CLIP and retrieve the most visually
        similar photos. Returns [(photo_id, similarity_score), ...].
        """
        tokens = self.tokenizer([query]).to(self.device)
        with self.torch.no_grad():
            text_emb = self.model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        results = self.collection.query(
            query_embeddings=[text_emb.squeeze().cpu().tolist()],
            n_results=min(n_results, self.collection.count()),
            include=["distances"],
        )
        return [
            (pid, 1.0 - dist)
            for pid, dist in zip(
                results["ids"][0], results["distances"][0]
            )
        ]

    def image_search(
        self, photo_path: Path, n_results: int = 50
    ) -> list[tuple[str, float]]:
        """
        Find photos that look visually similar to a given photo.
        Returns [(photo_id, similarity_score), ...].
        """
        img = self.preprocess(Image.open(photo_path).convert("RGB"))
        img_tensor = img.unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            img_emb = self.model.encode_image(img_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        results = self.collection.query(
            query_embeddings=[img_emb.squeeze().cpu().tolist()],
            n_results=min(n_results, self.collection.count()),
            include=["distances"],
        )
        return [
            (pid, 1.0 - dist)
            for pid, dist in zip(
                results["ids"][0], results["distances"][0]
            )
        ]

    @property
    def is_available(self) -> bool:
        try:
            return self.collection.count() > 0
        except Exception:
            return False
