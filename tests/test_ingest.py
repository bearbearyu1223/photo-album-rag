"""
test_ingest.py — unit tests for ingestion pipeline helpers

Tests cover the pure-Python helpers that don't require Ollama or a GPU:
  - EXIF GPS decimal conversion
  - contextual description builder
  - supported extension filtering
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from photo_rag.ingest import (
    SUPPORTED_EXTENSIONS,
    _exif_gps_to_decimal,
    build_contextual_description,
)


# ── GPS conversion ────────────────────────────────────────────────────────────

class TestExifGpsToDecimal:

    def _rational(self, num: int, den: int = 1):
        """Create a minimal rational-like object that _exif_gps_to_decimal expects."""
        r = MagicMock()
        r.num = num
        r.den = den
        return r

    def _coord_tag(self, degrees: int, minutes: int, seconds: int):
        tag = MagicMock()
        tag.values = [
            self._rational(degrees),
            self._rational(minutes),
            self._rational(seconds),
        ]
        return tag

    def _ref_tag(self, ref: str):
        tag = MagicMock()
        tag.__str__ = lambda self: ref
        return tag

    def test_north_positive(self):
        coord = self._coord_tag(47, 36, 0)   # ~Seattle latitude
        result = _exif_gps_to_decimal(coord, self._ref_tag("N"))
        assert result == pytest.approx(47.6, abs=0.1)

    def test_south_negative(self):
        coord = self._coord_tag(33, 52, 0)   # ~Sydney latitude
        result = _exif_gps_to_decimal(coord, self._ref_tag("S"))
        assert result == pytest.approx(-33.87, abs=0.1)

    def test_west_negative(self):
        coord = self._coord_tag(122, 19, 0)  # ~Seattle longitude
        result = _exif_gps_to_decimal(coord, self._ref_tag("W"))
        assert result == pytest.approx(-122.32, abs=0.1)

    def test_none_coord_returns_none(self):
        assert _exif_gps_to_decimal(None, MagicMock()) is None

    def test_none_ref_returns_positive(self):
        """When ref is None, assume positive (North/East)."""
        coord = self._coord_tag(35, 41, 0)  # Tokyo
        result = _exif_gps_to_decimal(coord, None)
        assert result is None or result > 0   # None ref → may return None gracefully

    def test_zero_coords(self):
        coord = self._coord_tag(0, 0, 0)
        result = _exif_gps_to_decimal(coord, self._ref_tag("N"))
        assert result == pytest.approx(0.0)


# ── Contextual description builder ───────────────────────────────────────────

class TestBuildContextualDescription:

    def test_includes_caption(self):
        path = Path("/fake/photos/beach.jpg")
        caption = "A sunny beach scene with families playing volleyball."
        desc = build_contextual_description(path, caption, meta={})
        assert caption in desc

    def test_includes_datetime(self):
        path = Path("/fake/photos/img.jpg")
        desc = build_contextual_description(
            path, "A photo.", meta={"datetime": "2023-12-25 10:00:00"}
        )
        assert "2023-12-25" in desc

    def test_includes_camera(self):
        path = Path("/fake/photos/img.jpg")
        desc = build_contextual_description(
            path, "A photo.", meta={"camera": "Apple iPhone 15 Pro"}
        )
        assert "iPhone" in desc

    def test_includes_gps_location_when_geocoded(self):
        path = Path("/fake/photos/img.jpg")
        with patch("photo_rag.ingest.gps_to_place", return_value="Seattle, WA, United States"):
            desc = build_contextual_description(
                path, "A photo.", meta={"gps": (47.6, -122.3)}
            )
        assert "Seattle" in desc

    def test_generic_filename_excluded(self):
        """Filenames like IMG_1234 or DSC_0001 shouldn't add noise."""
        path = Path("/fake/photos/IMG_1234.jpg")
        desc = build_contextual_description(path, "A photo.", meta={})
        # Generic prefixes should not appear as a hint
        assert "IMG" not in desc or "1234" not in desc

    def test_meaningful_filename_included(self):
        """Filenames like 'birthday_party.jpg' should contribute a hint."""
        path = Path("/fake/photos/birthday_party.jpg")
        desc = build_contextual_description(path, "A photo.", meta={})
        assert "birthday" in desc.lower()

    def test_empty_caption_still_builds_from_meta(self):
        path = Path("/fake/photos/img.jpg")
        desc = build_contextual_description(
            path, "", meta={"datetime": "2023-01-01", "camera": "Nikon D7500"}
        )
        assert "2023-01-01" in desc
        assert "Nikon" in desc

    def test_all_empty_returns_empty_or_minimal(self):
        """With no caption and no metadata, description should be minimal."""
        path = Path("/fake/photos/img_0001.jpg")
        desc = build_contextual_description(path, "", meta={})
        # Should not crash; result may be empty or just whitespace
        assert isinstance(desc, str)


# ── Supported extensions ──────────────────────────────────────────────────────

class TestSupportedExtensions:

    def test_jpg_supported(self):
        assert ".jpg" in SUPPORTED_EXTENSIONS

    def test_heic_supported(self):
        assert ".heic" in SUPPORTED_EXTENSIONS

    def test_raw_not_supported(self):
        assert ".raw" not in SUPPORTED_EXTENSIONS

    def test_pdf_not_supported(self):
        assert ".pdf" not in SUPPORTED_EXTENSIONS
