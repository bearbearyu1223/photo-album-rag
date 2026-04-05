"""
test_retrieval.py — unit tests for the hybrid retrieval pipeline

Tests are structured in three layers:
  1. Pure-logic tests (RRF, BM25 scoring) — no mocks needed
  2. Integration tests for PhotoRetriever — mock heavy dependencies
  3. Smoke tests for the full search pipeline end-to-end
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rank_bm25 import BM25Okapi

from photo_rag.retrieve import PhotoResult, PhotoRetriever


# ── 1. Pure logic: RRF ────────────────────────────────────────────────────────

class TestReciprocalRankFusion:
    """RRF is pure math — no I/O, no mocks needed."""

    def _make_result(self, photo_id: str, score: float = 0.0) -> PhotoResult:
        return PhotoResult(
            photo_id=photo_id,
            path=f"/fake/{photo_id}.jpg",
            description="",
            caption="",
            datetime="",
            camera="",
            rrf_score=score,
        )

    def test_single_list_preserves_order(self):
        results = [self._make_result(f"p{i}") for i in range(5)]
        fused = PhotoRetriever._reciprocal_rank_fusion(results)
        ids = [r.photo_id for r in fused]
        assert ids == ["p0", "p1", "p2", "p3", "p4"]

    def test_two_lists_boosts_overlap(self):
        """A photo ranked #1 in both lists should beat a photo ranked #1 in only one."""
        list_a = [self._make_result("shared"), self._make_result("only_a")]
        list_b = [self._make_result("shared"), self._make_result("only_b")]
        fused = PhotoRetriever._reciprocal_rank_fusion(list_a, list_b)

        ids = [r.photo_id for r in fused]
        assert ids[0] == "shared", "Photo ranked #1 in both lists should win"

    def test_deduplication(self):
        """A photo appearing in both lists should appear only once in output."""
        list_a = [self._make_result("dup"), self._make_result("a")]
        list_b = [self._make_result("dup"), self._make_result("b")]
        fused = PhotoRetriever._reciprocal_rank_fusion(list_a, list_b)
        ids = [r.photo_id for r in fused]
        assert ids.count("dup") == 1

    def test_scores_are_positive(self):
        results = [self._make_result(f"p{i}") for i in range(3)]
        fused = PhotoRetriever._reciprocal_rank_fusion(results)
        assert all(r.rrf_score > 0 for r in fused)

    def test_three_lists_fused(self):
        """RRF should accept any number of input lists."""
        a = [self._make_result("winner"), self._make_result("a2")]
        b = [self._make_result("winner"), self._make_result("b2")]
        c = [self._make_result("winner"), self._make_result("c2")]
        fused = PhotoRetriever._reciprocal_rank_fusion(a, b, c)
        assert fused[0].photo_id == "winner"

    def test_k_parameter_scores_decrease_with_rank(self):
        """Each successive rank should have a strictly lower RRF score."""
        results = [self._make_result(f"p{i}") for i in range(5)]
        fused = PhotoRetriever._reciprocal_rank_fusion(results, k=60)
        scores = [r.rrf_score for r in fused]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], (
                f"Score at rank {i} ({scores[i]}) should exceed rank {i+1} ({scores[i+1]})"
            )

    def test_k_formula_matches_expected(self):
        """RRF score for rank 1 with k=60 should be exactly 1/61."""
        results = [self._make_result("only")]
        fused = PhotoRetriever._reciprocal_rank_fusion(results, k=60)
        assert fused[0].rrf_score == pytest.approx(1 / 61)

    def test_empty_list_returns_empty(self):
        assert PhotoRetriever._reciprocal_rank_fusion([]) == []

    def test_multiple_empty_lists(self):
        assert PhotoRetriever._reciprocal_rank_fusion([], []) == []


# ── 2. Pure logic: BM25 scoring ───────────────────────────────────────────────

class TestBM25Scoring:
    """Validate BM25 scoring behavior independent of the retriever."""

    @pytest.fixture()
    def bm25(self):
        corpus = [
            "beach summer sunset ocean waves dolphin".split(),
            "mountain hiking trail forest trees elk".split(),
            "city skyline buildings night lights taxi".split(),
            "desert cactus sand dunes camel nomad".split(),
        ]
        return BM25Okapi(corpus), corpus

    def test_exact_match_scores_higher(self, bm25):
        bm25_model, _ = bm25
        beach_scores = bm25_model.get_scores("beach".split())
        # Only doc 0 contains "beach" — should be > 0
        assert beach_scores[0] > 0
        # Other docs don't have "beach" — should score 0
        assert beach_scores[1] == 0.0
        assert beach_scores[2] == 0.0
        assert beach_scores[3] == 0.0

    def test_multi_term_boosts_more_matches(self, bm25):
        bm25_model, _ = bm25
        scores_one = bm25_model.get_scores("beach".split())
        scores_two = bm25_model.get_scores("beach summer".split())
        # "beach summer" should boost doc 0 (has both) vs doc 3 (has beach + summer)
        # both should score higher than with single term
        assert max(scores_two) >= max(scores_one)

    def test_unknown_term_scores_zero(self, bm25):
        bm25_model, _ = bm25
        scores = bm25_model.get_scores("xyzzy".split())
        assert all(s == 0.0 for s in scores)


# ── 3. PhotoResult dataclass ──────────────────────────────────────────────────

class TestPhotoResult:

    def test_construction(self):
        r = PhotoResult(
            photo_id="/path/to/photo.jpg",
            path="/path/to/photo.jpg",
            description="A sunny beach day.",
            caption="Waves crashing on the shore.",
            datetime="2023-07-15 14:30:00",
            camera="iPhone 15 Pro",
        )
        assert r.rrf_score == 0.0
        assert r.rerank_score == 0.0
        assert r.caption == "Waves crashing on the shore."

    def test_score_assignment(self):
        r = PhotoResult(
            photo_id="p1", path="p1.jpg", description="",
            caption="", datetime="", camera="",
        )
        r.rrf_score = 0.42
        r.rerank_score = 1.7
        assert r.rrf_score == 0.42
        assert r.rerank_score == 1.7


# ── 4. PhotoRetriever with mocked dependencies ────────────────────────────────

class TestPhotoRetrieverBM25:
    """
    Test BM25 retrieval with a real BM25Okapi index but mocked
    ChromaDB and sentence-transformers to keep tests fast and offline.
    """

    @pytest.fixture()
    def retriever(self, fake_index_dir: Path) -> PhotoRetriever:
        """
        Build a PhotoRetriever with heavy deps mocked out.
        ChromaDB query/get calls return predictable fake metadata.
        """
        fake_descriptions = [
            "Photo at Seattle WA beach during summer family barbecue.",
            "Photo in Tokyo Japan cherry blossom park spring.",
            "Photo in Paris France Eiffel Tower sunset golden hour.",
        ]
        photo_ids = [f"/fake/photos/photo_{i}.jpg" for i in range(3)]

        # Mock ChromaDB collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [photo_ids],
            "documents": [fake_descriptions],
            "metadatas": [[
                {"path": pid, "caption": desc[:40], "datetime": "2023-01-01", "camera": "iPhone"}
                for pid, desc in zip(photo_ids, fake_descriptions)
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        }
        mock_collection.get.return_value = {
            "ids": photo_ids,
            "documents": fake_descriptions,
            "metadatas": [
                {"path": pid, "caption": desc[:40], "datetime": "2023-01-01", "camera": "iPhone"}
                for pid, desc in zip(photo_ids, fake_descriptions)
            ],
        }

        with (
            patch("photo_rag.retrieve.SentenceTransformer") as mock_st,
            patch("photo_rag.retrieve.CrossEncoder") as mock_ce,
            patch("photo_rag.retrieve.PersistentClient") as mock_chroma,
        ):
            # SentenceTransformer: encode() returns a fake 384-dim vector
            import numpy as np
            mock_embedder = MagicMock()
            mock_embedder.encode.return_value = np.array([0.1] * 384)
            mock_st.return_value = mock_embedder

            # CrossEncoder: predict() returns descending scores (best first)
            mock_reranker = MagicMock()
            mock_reranker.predict.side_effect = lambda pairs: [
                1.0 - (i * 0.1) for i in range(len(pairs))
            ]
            mock_ce.return_value = mock_reranker

            # ChromaDB
            mock_chroma_instance = MagicMock()
            mock_chroma_instance.get_collection.return_value = mock_collection
            mock_chroma.return_value = mock_chroma_instance

            r = PhotoRetriever(index_dir=fake_index_dir)

        return r

    def test_bm25_search_returns_results(self, retriever: PhotoRetriever):
        results = retriever._bm25_search("beach summer")
        assert len(results) > 0

    def test_bm25_beach_query_finds_beach_photo(self, retriever: PhotoRetriever):
        results = retriever._bm25_search("beach")
        ids = [r.photo_id for r in results]
        # photo_0 has "beach" in description — should appear
        assert "/fake/photos/photo_0.jpg" in ids

    def test_bm25_japan_query_finds_tokyo_photo(self, retriever: PhotoRetriever):
        results = retriever._bm25_search("Japan")
        ids = [r.photo_id for r in results]
        assert "/fake/photos/photo_1.jpg" in ids

    def test_search_returns_at_most_top_k(self, retriever: PhotoRetriever):
        results = retriever.search("beach", top_k=2)
        assert len(results) <= 2

    def test_search_returns_photo_results(self, retriever: PhotoRetriever):
        results = retriever.search("photo")
        assert all(isinstance(r, PhotoResult) for r in results)

    def test_rerank_scores_populated(self, retriever: PhotoRetriever):
        results = retriever.search("beach", top_k=3)
        assert all(r.rerank_score != 0.0 for r in results)

    def test_results_sorted_by_rerank_score(self, retriever: PhotoRetriever):
        results = retriever.search("beach", top_k=3)
        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True)
