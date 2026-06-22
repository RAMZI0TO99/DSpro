import os
import gc
import cv2
import torch
import easyocr
import whisperx
import librosa
from PIL import Image
from scenedetect import detect, ContentDetector
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding


class LocalHotIngestionPipeline:
    def __init__(self, clip_model, clip_tokenizer, clip_transform, db_client):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[BOOT-INGESTION] Initializing Local Hot Ingestion on {self.device}...")

        self.db_client = db_client
        self._ensure_collection_exists()

        self.bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.clip_transform = clip_transform

        print("[BOOT-INGESTION] Docking WhisperX (Base) into VRAM...")
        self.audio_model = whisperx.load_model("base", device=self.device, compute_type="float16")

        # FIX 2: Don't pre-load a single align model at boot.
        # Language is detected per-video at transcription time.
        # We cache loaded align models here so we never reload for the same language.
        self._align_model_cache = {}

        print("[BOOT-INGESTION] Docking EasyOCR into VRAM...")
        # verbose=False suppresses the progress bar that uses Unicode block chars (█)
        # which crash on Windows cp1252 console encoding. Model still downloads silently.
        self.ocr_reader = easyocr.Reader(
            ['en', 'ar'], 
            gpu=True, 
            verbose=False,
            # Use absolute path so it works both locally and inside Docker.
            # In Docker, models are baked into /root/.EasyOCR/model via COPY.
            # Locally, the EASYOCR_HOME env var handles the path via easyocr internals.
            model_storage_directory=os.environ.get("EASYOCR_HOME", os.path.abspath("./models/easyocr")) + "/model"
        )

        print("[BOOT-INGESTION COMPLETE] Ingestion Engine is HOT.\n")

    def _get_align_model(self, language_code: str):
        """Returns a cached alignment model for the given language, loading if needed."""
        if language_code not in self._align_model_cache:
            print(f"[ALIGN] Loading alignment model for language: '{language_code}'")
            try:
                align_model, align_meta = whisperx.load_align_model(
                    language_code=language_code, device=self.device
                )
                self._align_model_cache[language_code] = (align_model, align_meta)
            except Exception as e:
                # If alignment model is not available for this language, fall back to English
                print(f"[ALIGN] No model for '{language_code}', falling back to 'en'. Error: {e}")
                if "en" not in self._align_model_cache:
                    align_model, align_meta = whisperx.load_align_model(
                        language_code="en", device=self.device
                    )
                    self._align_model_cache["en"] = (align_model, align_meta)
                return self._align_model_cache["en"]
        return self._align_model_cache[language_code]

    def _ensure_collection_exists(self):
        """Creates the Qdrant collection only if it doesn't already exist."""
        collection_name = "video_segments"
        if not self.db_client.collection_exists(collection_name=collection_name):
            self.db_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "visual": models.VectorParams(size=768, distance=models.Distance.COSINE),
                    "audio": models.VectorParams(size=768, distance=models.Distance.COSINE),
                },
                sparse_vectors_config={
                    "text_sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                }
            )

    def is_video_indexed(self, video_id: str) -> bool:
        """Fast DB lookup to see if this video is already ingested."""
        try:
            records, _ = self.db_client.scroll(
                collection_name="video_segments",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=video_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            return len(records) > 0
        except Exception:
            return False

    def process_video_stream(self, video_path: str, video_id: str):
        """Yields SSE progress strings while ingesting a video end-to-end."""

        # --- IDEMPOTENCY CHECK ---
        yield f"data: [0/4] Checking database for '{video_id}'...\n\n"

        if self.is_video_indexed(video_id):
            yield f"data: [COMPLETE] '{video_id}' is already indexed. Skipping ingestion!\n\n"
            return

        # ----------------------------------------------------------------
        # PHASE 1: AUDIO TRANSCRIPTION (with auto language detection)
        # ----------------------------------------------------------------
        yield f"data: [1/4] Transcribing audio for '{video_id}'...\n\n"

        audio, _ = librosa.load(video_path, sr=16000)
        result = self.audio_model.transcribe(audio, batch_size=4)

        # FIX 2: Use the detected language for alignment instead of hardcoded 'en'
        detected_lang = result.get("language", "en")
        yield f"data: [1/4] Detected language: '{detected_lang}'. Aligning...\n\n"

        align_model, align_metadata = self._get_align_model(detected_lang)
        aligned_result = whisperx.align(
            result["segments"], align_model, align_metadata,
            audio, self.device, return_char_alignments=False
        )
        audio_segments = aligned_result["segments"]

        torch.cuda.empty_cache()
        gc.collect()

        # ----------------------------------------------------------------
        # PHASE 2: SCENE DETECTION (max 10s chunks)
        # ----------------------------------------------------------------
        yield "data: [2/4] Detecting scene boundaries (10s chunks)...\n\n"

        scene_list = detect(video_path, ContentDetector(threshold=27.0))
        raw_boundaries = [{"start": s[0].get_seconds(), "end": s[1].get_seconds()} for s in scene_list]

        if not raw_boundaries:
            cap = cv2.VideoCapture(video_path)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps_cap = cap.get(cv2.CAP_PROP_FPS)
            raw_boundaries.append({"start": 0.0, "end": total_frames / fps_cap if fps_cap > 0 else 0.0})
            cap.release()

        scene_segments = []
        max_duration = 10.0
        for scene in raw_boundaries:
            dur = scene["end"] - scene["start"]
            if dur > max_duration:
                chunks = int((dur // max_duration) + 1)
                for i in range(chunks):
                    scene_segments.append({
                        "start": scene["start"] + (i * (dur / chunks)),
                        "end": min(scene["start"] + ((i + 1) * (dur / chunks)), scene["end"])
                    })
            else:
                scene_segments.append(scene)

        yield f"data: [2/4] Found {len(scene_segments)} scene chunks.\n\n"

        # ----------------------------------------------------------------
        # PHASE 3: VISUAL OCR (every ~5 seconds per scene)
        # ----------------------------------------------------------------
        yield "data: [3/4] Running visual OCR scan...\n\n"

        scene_ocr_data = {}
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        for idx, scene in enumerate(scene_segments):
            scene_text_elements = []
            start_frame = int(scene['start'] * fps)
            end_frame = int(scene['end'] * fps)
            step = max(1, int(5 * fps))

            for frame_idx in range(start_frame, end_frame, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                scene_text_elements.extend(self.ocr_reader.readtext(gray, detail=0))

            scene_ocr_data[idx] = " ".join(list(set(scene_text_elements)))

        torch.cuda.empty_cache()
        gc.collect()

        # ----------------------------------------------------------------
        # PHASE 4: VECTOR GENERATION & DB UPSERT
        # ----------------------------------------------------------------
        yield f"data: [4/4] Generating vectors for {len(scene_segments)} chunks...\n\n"

        points_to_upsert = []

        for idx, scene in enumerate(scene_segments):
            start_frame = int(scene["start"] * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_embeddings = []

            # Sample 3 frames per scene for the visual vector
            for i in range(3):
                ret, frame = cap.read()
                if not ret:
                    break
                # FIX 1: Convert numpy array (OpenCV) → PIL Image before clip_transform
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_in = self.clip_transform(pil_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.clip_model.encode_image(img_in)
                    frame_embeddings.append((feat / feat.norm(dim=-1, keepdim=True)).cpu())

            vis_vec = torch.cat(frame_embeddings, dim=0).mean(dim=0).tolist() if frame_embeddings else [0.0] * 768
            # Build transcript from overlapping audio segments + OCR text
            speech = [
                s['text'] for s in audio_segments
                if (s['start'] <= scene['end'] and s['end'] >= scene['start'])
            ]
            transcript = f"{' '.join(speech)} {scene_ocr_data.get(idx, '')}".strip()

            if transcript:
                with torch.no_grad():
                    # FIX 4: Increased truncation from 70 → 200 chars.
                    # CLIP tokenizer handles the hard 77-token limit internally.
                    txt_in = self.clip_tokenizer([transcript[:200]]).to(self.device)
                    aud_vec = self.clip_model.encode_text(txt_in).cpu()[0].tolist()
                sparse_res = list(self.bm25_model.embed([transcript]))[0]
                sparse_vec = models.SparseVector(
                    indices=sparse_res.indices.tolist(),
                    values=sparse_res.values.tolist()
                )
            else:
                aud_vec = [0.0] * 768
                sparse_vec = models.SparseVector(indices=[], values=[])

            points_to_upsert.append(
                models.PointStruct(
                    id=hash(f"{video_id}_{scene['start']}") & 0xFFFFFFFFFFFFFFFF,
                    vector={"visual": vis_vec, "audio": aud_vec, "text_sparse": sparse_vec},
                    payload={
                        "video_id": video_id,
                        "start_timestamp": scene["start"],
                        "end_timestamp": scene["end"],
                        "transcript": transcript
                    }
                )
            )

            # Yield granular progress every 5 chunks
            if (idx + 1) % 5 == 0 or (idx + 1) == len(scene_segments):
                yield f"data: [4/4] Vectorized {idx + 1}/{len(scene_segments)} chunks...\n\n"

        cap.release()

        if points_to_upsert:
            self.db_client.upsert(collection_name="video_segments", points=points_to_upsert)

        yield f"data: [COMPLETE] '{video_id}' successfully indexed — {len(points_to_upsert)} chunks stored.\n\n"