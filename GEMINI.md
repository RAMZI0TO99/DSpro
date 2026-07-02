# DSpro Acoustic Search MODALITY & CLAP ARCHITECTURE

This document explains the upgrade from text-based BGE transcript embeddings to **true acoustic feature vectors** using the **LAION-CLAP** model (`laion/clap-htsat-unfused`), supported by our isolated audio extractor worker.

---

## 1. Context & Architectural Upgrade

### Legacy Approach (Monomer)
*   Previously, the `"audio"` named vector in Qdrant indexed BGE embeddings of the speech transcript (text).
*   This was a monomer approach that duplicated the textual information already captured by the dense text embedding and sparse BM25 vectors.

### New Approach (Authentic Acoustic Retrieval)
*   The `"audio"` named vector now holds **dense acoustic embeddings** of size **512** using Cosine similarity (`Distance.COSINE`).
*   Instead of duplicating the transcript text, the system extracts the physical audio clip for each `VideoChunk` and processes it through a pre-trained **LAION-CLAP** model.
*   This allows the system to analyze **raw sound signatures**, including:
    *   **Speaker Tone & Emotion:** Voice pitch, speed, and volume characteristics.
    *   **Environmental Backgrounds:** Music, background traffic, wind, hums, and ambient noise signatures.
    *   **Sound Anomalies:** Detections of gunshots, glass shattering, screams, alarms, sirens, and silence blocks.

---

## 2. Decoupled Acoustic Extractor Worker

To maintain clean baseline code and simplify database migration, all acoustic extraction logic is decoupled into an isolated service module:
*   **Path:** `app/services/audio_extractor.py`
*   **Class:** `AcousticExtractor`
*   **Features:**
    *   Lazy loading of `transformers.ClapModel` and `transformers.ClapProcessor`.
    *   Audio loading at **48,000 Hz** (required by CLAP).
    *   Audio track caching to optimize sequential scene chunk slicing.
    *   Zero-padding (`[0.0] * 512`) for documents, images, and non-audio sources to ensure Qdrant schema validation.

---

## 3. Running the Audio Database Migration

Because Qdrant does not support changing the vector dimensions of an existing named vector in-place, the collection must be recreated. A migration script has been provided to carry out this transition without data loss:

```bash
python migrate_audio_embeddings.py
```

### What the Migration Script Does:
1.  **Reads Existing State:** Fetches all points from the active Qdrant collection, caching their payloads, visual vectors, and sparse lexical vectors.
2.  **Re-Creates Collection:** Drops and recreates the collection, changing the `"audio"` named vector configuration from 384 dimensions to **512 dimensions**.
3.  **Extracts Acoustic Features:** Iterates over the cached points, locates the original media file in the `media/` directory, loads the audio using `AcousticExtractor` at 48kHz, slices the audio segment, runs the CLAP audio encoder to extract the 512-d acoustic embedding, and re-inserts the updated point back to Qdrant.
