"""
conftest.py — shared pytest fixtures for photo_rag tests
"""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Minimal fake index fixture ────────────────────────────────────────────────

@pytest.fixture()
def fake_index_dir(tmp_path: Path) -> Path:
    """
    Creates a minimal on-disk index that PhotoRetriever can load,
    with all heavy dependencies (ChromaDB, sentence-transformers,
    cross-encoder) replaced by lightweight mocks.
    """
    index_dir = tmp_path / "photo_index"
    (index_dir / "chroma").mkdir(parents=True)

    # config.json
    config = {
        "embedding_model": "all-MiniLM-L6-v2",
        "vision_model": "llava",
        "clip_enabled": False,
    }
    (index_dir / "config.json").write_text(json.dumps(config))

    # bm25_state.pkl — three fake photos
    from rank_bm25 import BM25Okapi

    descriptions = [
        "Photo taken in Seattle WA. A family at the beach during summer.",
        "Photo taken in Tokyo Japan. Cherry blossom trees in a park in spring.",
        "Photo taken in Paris France. The Eiffel Tower at sunset with tourists.",
    ]
    photo_ids = [f"/fake/photos/photo_{i}.jpg" for i in range(3)]
    corpus = [d.lower().split() for d in descriptions]
    bm25 = BM25Okapi(corpus)

    state = {
        "corpus": corpus,
        "ids": photo_ids,
        "id_to_path": {pid: pid for pid in photo_ids},
        "bm25": bm25,
    }
    with open(index_dir / "bm25_state.pkl", "wb") as f:
        pickle.dump(state, f)

    return index_dir


@pytest.fixture()
def fake_descriptions() -> list[str]:
    return [
        "Date: 2023-07-04. Location: Seattle, WA. A family barbecue on the 4th of July.",
        "Date: 2023-03-28. Location: Tokyo, Japan. Cherry blossoms in Ueno Park.",
        "Date: 2023-09-15. Location: Paris, France. The Eiffel Tower at golden hour.",
    ]


@pytest.fixture()
def fake_photo_ids(fake_descriptions) -> list[str]:
    return [f"/fake/photos/photo_{i}.jpg" for i in range(len(fake_descriptions))]
