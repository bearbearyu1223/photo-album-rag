"""
ingest.py — Photo Album RAG Ingestion Pipeline
================================================
Walks a photo directory, extracts EXIF metadata, generates captions
via a local vision model (LLaVA through Ollama), builds contextual
descriptions, and indexes everything into ChromaDB + BM25.

Usage:
    python ingest.py --photos ~/Pictures --index ./photo_index

Dependencies:
    pip install chromadb rank_bm25 sentence-transformers pillow \
                exifread geopy ollama tqdm
    ollama pull llava
"""

import argparse
import base64
import json
import os
import pickle
import re
from pathlib import Path

import exifread
import ollama
from chromadb import PersistentClient
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from PIL import Image
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # swap for BAAI/bge-small-en-v1.5 if you want
VISION_MODEL = "llava"                  # or "moondream" for faster/lighter captions
CAPTION_PROMPT = (
    "Describe this photo in 2-3 sentences. Focus on: who or what is in it, "
    "what activity or moment is happening, the setting or environment, and the mood. "
    "Be specific. Do not say 'the image shows' — just describe directly."
)

# ── EXIF Helpers ─────────────────────────────────────────────────────────────

def extract_exif(photo_path: Path) -> dict:
    """Return a dict of useful EXIF fields from a photo."""
    meta = {}
    try:
        with open(photo_path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="GPS GPSLongitude", details=False)

        # Date/time
        for key in ("EXIF DateTimeOriginal", "Image DateTime"):
            if key in tags:
                meta["datetime"] = str(tags[key])
                break

        # Camera
        if "Image Make" in tags:
            make = str(tags["Image Make"]).strip()
            model = str(tags.get("Image Model", "")).strip()
            meta["camera"] = f"{make} {model}".strip()

        # GPS
        lat = _exif_gps_to_decimal(
            tags.get("GPS GPSLatitude"), tags.get("GPS GPSLatitudeRef")
        )
        lon = _exif_gps_to_decimal(
            tags.get("GPS GPSLongitude"), tags.get("GPS GPSLongitudeRef")
        )
        if lat and lon:
            meta["gps"] = (lat, lon)

    except Exception as e:
        print(f"  [EXIF] Could not read {photo_path.name}: {e}")

    return meta


def _exif_gps_to_decimal(coord_tag, ref_tag) -> float | None:
    """Convert EXIF GPS rational values to a signed decimal degree."""
    if coord_tag is None:
        return None
    try:
        vals = coord_tag.values
        degrees = float(vals[0].num) / float(vals[0].den)
        minutes = float(vals[1].num) / float(vals[1].den)
        seconds = float(vals[2].num) / float(vals[2].den)
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref_tag and str(ref_tag) in ("S", "W"):
            decimal = -decimal
        return decimal
    except Exception:
        return None


# ── Reverse Geocoding ─────────────────────────────────────────────────────────

_geocoder = Nominatim(user_agent="photo_rag_indexer")
_geo_cache: dict[tuple, str] = {}   # simple in-process cache

def gps_to_place(lat: float, lon: float) -> str:
    """Return a human-readable location string for GPS coordinates."""
    key = (round(lat, 3), round(lon, 3))   # round to ~100 m precision
    if key in _geo_cache:
        return _geo_cache[key]
    try:
        location = _geocoder.reverse(f"{lat}, {lon}", timeout=5, language="en")
        if location:
            addr = location.raw.get("address", {})
            parts = [
                addr.get("city") or addr.get("town") or addr.get("village"),
                addr.get("state"),
                addr.get("country"),
            ]
            place = ", ".join(p for p in parts if p)
            _geo_cache[key] = place
            return place
    except (GeocoderTimedOut, Exception):
        pass
    return ""


# ── Vision Model Caption ──────────────────────────────────────────────────────

def generate_caption(photo_path: Path) -> str:
    """
    Send the photo to a local LLaVA instance via Ollama and return a caption.
    Resizes large images first to keep latency reasonable.
    """
    # Resize to max 1024px on longest side to keep inference fast
    img = Image.open(photo_path).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)

    # Save to a temp buffer and base64-encode
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": CAPTION_PROMPT,
                "images": [img_b64],
            }],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"  [LLaVA] Caption failed for {photo_path.name}: {e}")
        return ""


# ── Contextual Description Builder ───────────────────────────────────────────

def build_contextual_description(
    photo_path: Path,
    caption: str,
    meta: dict,
) -> str:
    """
    Combine EXIF metadata and vision caption into a single rich text description.
    This is the 'contextual chunk' from the Anthropic blog — it gives the
    embedding model rich, disambiguating context rather than bare pixels.
    """
    lines = []

    # Date
    if "datetime" in meta:
        lines.append(f"Date taken: {meta['datetime']}.")

    # Location
    if "gps" in meta:
        place = gps_to_place(*meta["gps"])
        if place:
            lines.append(f"Location: {place}.")
        else:
            lines.append(f"GPS coordinates: {meta['gps'][0]:.4f}, {meta['gps'][1]:.4f}.")

    # Camera
    if "camera" in meta:
        lines.append(f"Camera: {meta['camera']}.")

    # Filename (often has useful hints like "birthday" or "beach")
    stem = photo_path.stem.replace("_", " ").replace("-", " ")
    if not re.match(r"^(img|dsc|photo|p\d+)\s", stem, re.IGNORECASE):
        lines.append(f"Filename hint: {stem}.")

    # The actual visual description from the vision model
    if caption:
        lines.append(caption)

    return " ".join(lines)


# ── Indexing ──────────────────────────────────────────────────────────────────

def build_index(photos_dir: Path, index_dir: Path, use_clip: bool = False):
    """
    Main ingestion loop. For every supported photo:
      1. Extract EXIF
      2. Generate caption
      3. Build contextual description
      4. Embed and store in ChromaDB
      5. Accumulate for BM25

    Saves the BM25 index and an ID→path mapping to disk alongside ChromaDB.
    """
    index_dir.mkdir(parents=True, exist_ok=True)


    # Optionally set up CLIP indexer
    clip_indexer = None
    if use_clip:
        try:
            from photo_rag.clip_search import CLIPIndexer
            print("Loading CLIP model...")
            clip_indexer = CLIPIndexer(index_dir)
        except ImportError as e:
            print(f"  [CLIP] Skipping: {e}")

    # Discover photos
    photo_paths = sorted([
        p for p in photos_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    print(f"Found {len(photo_paths)} photos in {photos_dir}\n")

    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Set up ChromaDB
    chroma = PersistentClient(path=str(index_dir / "chroma"))
    collection = chroma.get_or_create_collection(
        name="photos",
        metadata={"hnsw:space": "cosine"},
    )

    # Track already-indexed IDs so re-runs are incremental
    existing_ids = set(collection.get()["ids"])

    bm25_corpus: list[list[str]] = []   # tokenized descriptions for BM25
    bm25_ids: list[str] = []            # parallel list of photo IDs
    id_to_path: dict[str, str] = {}     # photo_id → absolute path string

    # Load existing BM25 data if present (for incremental re-indexing)
    bm25_state_path = index_dir / "bm25_state.pkl"
    if bm25_state_path.exists():
        with open(bm25_state_path, "rb") as f:
            saved = pickle.load(f)
        bm25_corpus = saved["corpus"]
        bm25_ids = saved["ids"]
        id_to_path = saved["id_to_path"]
        print(f"Loaded existing BM25 state with {len(bm25_ids)} entries.\n")

    # Ingest photos
    new_count = 0
    for photo_path in tqdm(photo_paths, desc="Indexing photos"):
        photo_id = str(photo_path.resolve())

        if photo_id in existing_ids:
            continue   # already indexed — skip

        # 1. EXIF
        meta = extract_exif(photo_path)

        # 2. Caption
        caption = generate_caption(photo_path)

        # 3. Contextual description
        description = build_contextual_description(photo_path, caption, meta)

        if not description.strip():
            continue

        # 4. Embed + store in ChromaDB
        embedding = embedder.encode(description).tolist()
        collection.add(
            ids=[photo_id],
            embeddings=[embedding],
            documents=[description],
            metadatas=[{
                "path": str(photo_path),
                "filename": photo_path.name,
                "datetime": meta.get("datetime", ""),
                "camera": meta.get("camera", ""),
                "caption": caption,
            }],
        )

        # 5. Accumulate for BM25
        tokens = description.lower().split()
        bm25_corpus.append(tokens)
        bm25_ids.append(photo_id)
        id_to_path[photo_id] = str(photo_path)


        # 6. CLIP image embedding (optional)
        if clip_indexer is not None:
            clip_indexer.add(photo_id, photo_path)

        new_count += 1

    print(f"\nIndexed {new_count} new photos. "
          f"Total in index: {len(bm25_ids)}.")

    # Rebuild and save BM25 index
    print("Building BM25 index...")
    bm25 = BM25Okapi(bm25_corpus)
    with open(bm25_state_path, "wb") as f:
        pickle.dump({
            "corpus": bm25_corpus,
            "ids": bm25_ids,
            "id_to_path": id_to_path,
            "bm25": bm25,
        }, f)

    # Save embedder model name for retrieval consistency
    config = {"embedding_model": EMBEDDING_MODEL, "vision_model": VISION_MODEL, "clip_enabled": use_clip}
    with open(index_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Index saved to {index_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Index a photo library for RAG.")
    parser.add_argument(
        "--photos", type=Path, required=True,
        help="Root directory of your photo library (e.g. ~/Pictures)",
    )
    parser.add_argument(
        "--index", type=Path, default=Path("./photo_index"),
        help="Where to store the index (default: ./photo_index)",
    )
    parser.add_argument("--clip", action="store_true", help="Also build CLIP image-embedding index (requires [clip] extra)")
    args = parser.parse_args()
    build_index(args.photos.expanduser(), args.index.expanduser(), use_clip=args.clip)


if __name__ == "__main__":
    main()
