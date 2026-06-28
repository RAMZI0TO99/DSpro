from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel
import asyncio
import os
import re
import shutil
import cv2
import json

from app.core.schemas import SearchRequest, ChatRequest, VideoMoveRequest
from app.core.config import llm_settings, save_settings, LLMSettings
from app.services.llm_service import llm_service
from app.services.graph_service import graph_service
from app.services import vector_search
import torch
from qdrant_client import models

# Helper to ensure models are loaded before use
def _ensure_models_loaded():
    vector_search._load_models()

# Shortcuts to vector_search module attributes
def get_qdrant_client():
    return vector_search.qdrant_client

def get_device():
    return vector_search.device

def get_clip_model():
    _ensure_models_loaded()
    return vector_search.clip_model

def get_clip_tokenizer():
    _ensure_models_loaded()
    return vector_search.clip_tokenizer

def get_bm25_model():
    _ensure_models_loaded()
    return vector_search.bm25_model

def get_ingestion_engine():
    _ensure_models_loaded()
    return vector_search.ingestion_engine

def get_dense_text_model():
    _ensure_models_loaded()
    return vector_search.dense_text_model

router = APIRouter()

# --- Upload limits ---
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB
ALLOWED_MIME_TYPES = {
    # Video
    "video/mp4", "video/x-matroska", "video/mkv",
    # Audio
    "audio/mpeg", "audio/wav", "audio/x-m4a",
    # Image
    "image/jpeg", "image/png", "image/webp",
    # Document
    "application/pdf"
}

# =====================================================================
# ROUTES
# =====================================================================

@router.get("/thumbnail/{video_id}")
async def get_thumbnail(video_id: str):
    """Extracts a representative frame from a video and returns it as a JPEG thumbnail."""
    import cv2
    from fastapi.responses import Response

    if not os.path.exists("media"):
        raise HTTPException(status_code=404, detail="Media directory not found.")

    # Find the media file for this video_id
    video_file = None
    for root, dirs, files in os.walk("media"):
        for filename in files:
            raw_name = os.path.splitext(filename)[0]
            candidate_id = re.sub(r'\W+', '_', raw_name.lower()).strip('_')
            if candidate_id == video_id:
                video_file = os.path.join(root, filename)
                break
        if video_file:
            break

    if not video_file:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")

    # --- Thumbnail Logic by Media Type ---
    ext = os.path.splitext(video_file)[1].lower()
    
    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        # If it's an image, just resize it
        frame = cv2.imread(video_file)
        if frame is None:
            raise HTTPException(status_code=500, detail="Could not read image file.")
    elif ext in [".mp3", ".wav", ".m4a"]:
        # For audio, return a completely black frame with a music note (or just black for now)
        import numpy as np
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        # Draw a simple blue box/text to represent audio
        cv2.putText(frame, "AUDIO", (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
    elif ext == ".pdf":
        # For PDFs, use PyMuPDF to render the first page
        import fitz
        import numpy as np
        try:
            doc = fitz.open(video_file)
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            # Convert pixmap to numpy array for cv2
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            # PyMuPDF produces RGB (or RGBA), cv2 expects BGR
            if pix.n == 4:
                frame = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error generating PDF thumbnail: {e}")
            frame = np.zeros((240, 240, 3), dtype=np.uint8)
            cv2.putText(frame, "PDF", (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    else:
        # Default Video handling
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video file.")

        # Seek to 5% into the video to skip intros/black frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = max(1, int(total_frames * 0.05))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise HTTPException(status_code=500, detail="Could not extract frame from video.")

    # Resize to a compact thumbnail size (width=240, maintain aspect ratio)
    h, w = frame.shape[:2]
    new_w = 240
    new_h = int(h * (new_w / w)) if w > 0 else 240
    if new_w > 0 and new_h > 0:
        frame = cv2.resize(frame, (new_w, new_h))

    # Encode as JPEG in memory
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@router.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the Vanilla HTML/JS Frontend."""
    with open("Frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()


# --- UPLOAD ENDPOINT ---
from fastapi import Form

@router.post("/upload")
async def upload_video(file: UploadFile = File(...), folder_path: str = Form(None)):
    """
    Saves an uploaded video to /media with MIME type and size validation.
    Optionally saves it inside a subdirectory if folder_path is provided.
    Accepts: MP4 and MKV. Max size: 2 GB.
    """
    # Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only MP4 and MKV are accepted."
        )

    if not os.path.exists("media"):
        os.makedirs("media")

    # Secure the filename to just its basename
    raw_filename = file.filename.replace("\\", "/")
    safe_basename = os.path.basename(raw_filename)
    
    # If folder_path is missing, try inferring from the client's raw filename
    if not folder_path or folder_path == "undefined":
        folder_path = os.path.dirname(raw_filename)

    target_dir = "media"
    if folder_path and folder_path != "undefined":
        # Allow alphanumeric, underscore, hyphen, space, and forward slash for nested folders
        clean_folder = re.sub(r'[^a-zA-Z0-9_\-\ /]', '', folder_path).strip()
        clean_folder = clean_folder.replace('//', '/').strip('/')
        if clean_folder:
            target_dir = os.path.join("media", clean_folder)
            
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, safe_basename).replace("\\", "/")
    bytes_written = 0
    chunk_size = 1024 * 1024  # 1 MB chunks

    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE:
                    buffer.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the 2 GB maximum allowed size."
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    # Generate the clean ID (matches ingestion + library logic)
    raw_name = os.path.splitext(safe_basename)[0]
    clean_id = re.sub(r'\W+', '_', raw_name.lower()).strip('_')

    return {
        "message": "Upload successful",
        "video_id": clean_id,
        "file_path": file_path,
        "size_bytes": bytes_written
    }


# --- BACKGROUND ML JOBS ---
active_ingestions = {}  # video_id -> {"last_message": str, "done": bool, "error": str, "canceled": bool}

def cleanup_canceled_video(video_id: str, file_path: str):
    """Deletes vectors from Qdrant and removes the file from disk if ingestion is canceled."""
    print(f"[CANCEL] Cleaning up resources for video {video_id}...")
    try:
        qdrant_client = get_qdrant_client()
        from qdrant_client import models
        qdrant_client.delete(
            collection_name="video_segments",
            points_selector=models.Filter(
                must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))]
            )
        )
    except Exception as e:
        print(f"[CANCEL] Error cleaning Qdrant for {video_id}: {e}")
        
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"[CANCEL] Deleted file: {file_path}")
    except Exception as e:
        print(f"[CANCEL] Error deleting file {file_path}: {e}")

def bg_ingest_task(video_id: str, file_path: str):
    """Runs the heavy ML ingestion loop in a separate thread."""
    active_ingestions[video_id] = {"last_message": "data: [0/4] Starting...\n\n", "done": False, "error": None, "canceled": False}
    print(f"[INGEST] Starting background job for '{video_id}'...")
    try:
        ingestion_engine = get_ingestion_engine()
        gen = ingestion_engine.process_media_stream(file_path, video_id)
        is_canceled = False
        
        for msg in gen:
            if video_id in active_ingestions and active_ingestions[video_id].get("canceled"):
                is_canceled = True
                break
                
            if video_id in active_ingestions:
                active_ingestions[video_id]["last_message"] = msg
                # Log the progress string to the server terminal
                clean_msg = msg.replace("data: ", "").strip()
                if clean_msg:
                    print(f"[INGEST '{video_id}'] {clean_msg}")
                    
        # Guarantee all file locks (cv2.VideoCapture) are dropped by closing the generator
        gen.close()
        
        if is_canceled:
            print(f"[INGEST] Job '{video_id}' was canceled by user.")
            active_ingestions[video_id]["last_message"] = "data: [CANCELED] Ingestion aborted.\n\n"
            active_ingestions[video_id]["done"] = True
            cleanup_canceled_video(video_id, file_path)
            return
                    
        if video_id in active_ingestions:
            print(f"[INGEST] Job '{video_id}' completed successfully.")
            active_ingestions[video_id]["done"] = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        if video_id in active_ingestions:
            print(f"[INGEST] Job '{video_id}' failed: {e}")
            active_ingestions[video_id]["last_message"] = f"data: [ERROR] {str(e)}\n\n"
            active_ingestions[video_id]["error"] = str(e)
            active_ingestions[video_id]["done"] = True

@router.post("/api/ingest/start")
def start_ingest_job(video_id: str, file_path: str, bg_tasks: BackgroundTasks):
    """Triggers the ingestion job in the background."""
    import urllib.parse
    file_path = urllib.parse.unquote(file_path)
    
    if video_id not in active_ingestions or active_ingestions[video_id]["done"]:
        bg_tasks.add_task(bg_ingest_task, video_id, file_path)
    return {"success": True, "video_id": video_id}

@router.get("/api/ingest/active")
def get_active_ingestions():
    """Returns a list of currently ingesting video IDs so the UI can re-attach on refresh."""
    active = [vid for vid, state in active_ingestions.items() if not state["done"]]
    return {"active_jobs": active}

@router.post("/api/ingest/{video_id}/cancel")
def cancel_ingest_job(video_id: str):
    """Signals an active ingestion loop to break early."""
    if video_id in active_ingestions:
        active_ingestions[video_id]["canceled"] = True
        return {"success": True}
    return {"success": False, "error": "Not found or not active"}

# --- INGEST STREAMING ENDPOINT ---
@router.get("/ingest/{video_id}")
async def ingest_video_stream(video_id: str, file_path: str = None):
    """Streams the progress of the ingestion engine by polling the active_ingestions dictionary."""
    async def event_generator():
        last_sent = None
        while True:
            if video_id not in active_ingestions:
                yield "data: [COMPLETE] No active ingestion.\n\n"
                break
                
            state = active_ingestions[video_id]
            
            if state["last_message"] != last_sent:
                last_sent = state["last_message"]
                yield last_sent
            else:
                # Keep-alive ping to prevent the connection from dropping (WinError 10054)
                yield ": keepalive\n\n"
                
            if state["done"]:
                yield "data: [COMPLETE] Ingestion finished.\n\n"
                # Clean up memory after a short delay so client can finish
                await asyncio.sleep(2)
                if video_id in active_ingestions:
                    del active_ingestions[video_id]
                break
                
            await asyncio.sleep(0.5)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- LIBRARY ENDPOINT ---
@router.get("/library")
def get_media_library():
    """Dynamically scans the media folder and returns all video IDs, titles, file sizes, and folders."""
    video_lib = {}
    title_lib = {}
    size_lib = {}
    folder_lib = {}
    media_dir = "media"

    if not os.path.exists(media_dir):
        return {"videoLibrary": {}, "titleLibrary": {}, "sizeLibrary": {}, "folderLibrary": {}}

    for root, dirs, files in os.walk(media_dir):
        for filename in files:
            ext = filename.lower().split('.')[-1]
            if ext in ["mp4", "mkv", "mp3", "wav", "m4a", "jpg", "jpeg", "png", "webp", "pdf"]:
                raw_name = os.path.splitext(filename)[0]
                clean_id = re.sub(r'\W+', '_', raw_name.lower()).strip('_')
                
                rel_path = os.path.relpath(os.path.join(root, filename), media_dir)
                folder_name = os.path.dirname(rel_path).replace('\\', '/')

                import urllib.parse
                encoded_path = "/".join(urllib.parse.quote(p) for p in rel_path.replace(os.sep, '/').split('/'))
                video_lib[clean_id] = f"/media/{encoded_path}"
                title_lib[clean_id] = raw_name.replace("_", " ")
                size_lib[clean_id] = os.path.getsize(os.path.join(root, filename))
                folder_lib[clean_id] = folder_name

    return {
        "videoLibrary": video_lib,
        "titleLibrary": title_lib,
        "sizeLibrary": size_lib,
        "folderLibrary": folder_lib
    }


# =====================================================================
# INTENT A: Search & Seek (Tri-Modal Vector Search)
# =====================================================================
@router.post("/search")
def search_video(request: SearchRequest):
    """Tri-modal hybrid search using CLIP visual + CLIP audio-text + BM25, fused with RRF."""
    try:
        _ensure_models_loaded()
        
        clip_model = get_clip_model()
        clip_tokenizer = get_clip_tokenizer()
        bm25_model = get_bm25_model()
        dense_text_model = get_dense_text_model()
        device = get_device()
        qdrant_client = get_qdrant_client()
        
        # 1. Intercept and Optimize with LLM
        optimized_query_raw = llm_service.optimize_query(request.query, llm_settings.provider, llm_settings.model)
        if "|" in optimized_query_raw:
            parts = optimized_query_raw.split("|")
            query_original = parts[0].strip()
            query_english = parts[1].strip()
        else:
            query_original = optimized_query_raw.strip()
            query_english = optimized_query_raw.strip()

        # 2. Generate Dense Semantic Vector (CLIP - English Only)
        with torch.no_grad():
            text_input = clip_tokenizer([query_english]).to(device)
            dense_tensor = clip_model.encode_text(text_input)
            dense_tensor /= dense_tensor.norm(dim=-1, keepdim=True)
            dense_vector = dense_tensor.tolist()[0]

        # 3. Generate Sparse Lexical Vector (BM25 - Original Language)
        sparse_result = list(bm25_model.embed([query_original]))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist()
        )

        # 3b. Generate Multilingual Dense Text Vector (Original Language)
        multilingual_dense_vector = list(dense_text_model.embed([query_original]))[0].tolist()

        # 4. Tri-Modal Hybrid Search (Sequential searches to preserve absolute scores)
        visual_res = qdrant_client.query_points(collection_name="video_segments", query=dense_vector, using="visual", limit=request.top_k, with_payload=True).points
        audio_res = qdrant_client.query_points(collection_name="video_segments", query=multilingual_dense_vector, using="audio", limit=request.top_k, with_payload=True).points
        sparse_res = qdrant_client.query_points(collection_name="video_segments", query=sparse_vector, using="text_sparse", limit=request.top_k, with_payload=True).points
        
        batch_res = [visual_res, audio_res, sparse_res]
        
        merged = {}
        for res_list in batch_res:
            for r in res_list:
                if r.id not in merged:
                    # BM25 scores can be > 1.0, cap them
                    r.score = min(r.score, 1.0)
                    merged[r.id] = r
                else:
                    merged[r.id].score = max(merged[r.id].score, min(r.score, 1.0))
                    
        # Sort by absolute score
        final_points = sorted(merged.values(), key=lambda x: x.score, reverse=True)[:request.top_k]

        output = []
        THRESHOLD = 0.3  # Confidence threshold (30%) to prevent weak matches

        for point in final_points:
            capped_score = min(point.score, 1.0)
            
            if capped_score < THRESHOLD:
                continue

            transcript = point.payload.get("transcript", "")
            # Return up to 500 chars for the result detail card
            snippet = (transcript[:500] + "...") if len(transcript) > 500 else transcript

            output.append({
                "video_id": point.payload.get("video_id"),
                "start_timestamp": point.payload.get("start_timestamp"),
                "end_timestamp": point.payload.get("end_timestamp"),
                "matched_transcript": snippet,
                "hybrid_rrf_score": round(capped_score, 4),
                "llm_optimized_query": query_original
            })

        if not output:
            return {"results": [], "message": "Out of domain query. No relevant video segments found."}

        return {"results": output}
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}


# =====================================================================
# INTENT B: Chat & Summarize (Context-Aware LLM generation)
# =====================================================================
@router.post("/chat")
def chat_with_video(request: ChatRequest):
    """Handles multi-turn Q&A with transparent 'Thought Streaming' and True Dynamic RAG."""
    _ensure_models_loaded()
    
    clip_model = get_clip_model()
    clip_tokenizer = get_clip_tokenizer()
    bm25_model = get_bm25_model()
    device = get_device()
    qdrant_client = get_qdrant_client()
    
    if not request.target_video_ids:
        return {"answer": "Error: No videos selected for chat."}

    def chat_flow():
        # --- THOUGHT PROCESS: STEP 1 ---
        yield "🧠 *Optimizing search query...*\n\n"
        
        # 1. Retrieve ALL chunks to see if we need RAG
        records = []
        for vid_id in request.target_video_ids:
            try:
                v_records, _ = qdrant_client.scroll(
                    collection_name="video_segments",
                    scroll_filter=models.Filter(must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=vid_id)
                        )
                    ]),
                    limit=1000,
                    with_payload=True,
                    with_vectors=False
                )
                records.extend(v_records)
            except IndexError:
                # Qdrant local backend throws IndexError if the collection is completely empty
                pass
            except Exception as e:
                print(f"Warning: Failed to scroll qdrant for video {vid_id}: {e}")

        if not records:
            yield "❌ *Error: No data found for the selected media in the database.*\n\n"
            return

        video_transcripts = {}
        for r in records:
            vid = r.payload.get("video_id")
            video_transcripts.setdefault(vid, []).append(r)
            
        media_lib = get_media_library()["videoLibrary"]
        folder_lib = get_media_library()["folderLibrary"]

        def format_transcript_data(v_transcripts, prefix=""):
            res = prefix + "\n"
            summaries = []
            for i, (vid, segs) in enumerate(v_transcripts.items(), 1):
                segs.sort(key=lambda x: x.payload.get("start_timestamp", 0))
                t_text = " ".join([s.payload.get("transcript", "") for s in segs])
                
                ext = media_lib.get(vid, "").split('.')[-1].lower()
                if ext == "pdf":
                    ftype = "PDF Document"
                elif ext in ["jpg", "jpeg", "png", "webp"]:
                    ftype = "Image"
                elif ext in ["mp3", "wav", "m4a"]:
                    ftype = "Audio File"
                elif ext in ["mp4", "mkv"]:
                    ftype = "Video"
                else:
                    ftype = "Media"
                    
                folder_name = folder_lib.get(vid, "root")
                if not folder_name or folder_name == ".":
                    folder_name = "root"
                    
                res += f"\n\n--- [File {i}] Name: {vid} | Type: {ftype} | Folder: {folder_name} ---\n{t_text}"
                summaries.append(f"{i}. {vid} ({ftype}) in folder '{folder_name}'")
            return res.strip(), summaries

        full_transcript, file_summaries = format_transcript_data(video_transcripts)

        # --- TRUE DYNAMIC RAG: CONTEXT LIMITING ---
        MAX_CHARS = 4000
        MAX_CONTEXT_CHARS = 24000 # ~6000 tokens safe limit
        
        if len(full_transcript) > MAX_CHARS:
            # We must use Semantic Search because the video is long
            optimized_query_raw = llm_service.optimize_query(request.query, llm_settings.provider, llm_settings.model)
            if "|" in optimized_query_raw:
                parts = optimized_query_raw.split("|")
                query_original = parts[0].strip()
                query_english = parts[1].strip()
            else:
                query_original = optimized_query_raw.strip()
                query_english = optimized_query_raw.strip()
            
            # --- THOUGHT PROCESS: STEP 2 ---
            yield "🔎 *Scanning vector database...*\n\n"
            
            with torch.no_grad():
                text_input = clip_tokenizer([query_english]).to(device)
                dense_tensor = clip_model.encode_text(text_input)
                dense_tensor /= dense_tensor.norm(dim=-1, keepdim=True)
                dense_vector = dense_tensor.tolist()[0]
                
            sparse_result = list(bm25_model.embed([query_original]))[0]
            sparse_vector = models.SparseVector(
                indices=sparse_result.indices.tolist(),
                values=sparse_result.values.tolist()
            )
            
            dense_text_model = get_dense_text_model()
            multilingual_dense_vector = list(dense_text_model.embed([query_original]))[0].tolist()
            
            video_filter = models.Filter(must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchAny(any=request.target_video_ids)
                )
            ])

            # Retrieve a large pool (50 chunks) to filter down (Sequential to preserve absolute scores)
            visual_res = qdrant_client.query_points(collection_name="video_segments", query=dense_vector, using="visual", query_filter=video_filter, limit=50, with_payload=True).points
            audio_res = qdrant_client.query_points(collection_name="video_segments", query=multilingual_dense_vector, using="audio", query_filter=video_filter, limit=50, with_payload=True).points
            sparse_res = qdrant_client.query_points(collection_name="video_segments", query=sparse_vector, using="text_sparse", query_filter=video_filter, limit=50, with_payload=True).points
            
            batch_res = [visual_res, audio_res, sparse_res]
            
            merged = {}
            for res_list in batch_res:
                for r in res_list:
                    if r.id not in merged:
                        r.score = min(r.score, 1.0)
                        merged[r.id] = r
                    else:
                        merged[r.id].score = max(merged[r.id].score, min(r.score, 1.0))
                        
            final_points = sorted(merged.values(), key=lambda x: x.score, reverse=True)[:50]
            
            # Filter and dynamically pack chunks
            valid_chunks = []
            current_chars = 0
            
            for r in final_points:
                score = min(r.score, 1.0)  # Cap score to 1.0 for consistency
                if score < 0.3:
                    continue # Skip low relevance chunks
                    
                chunk_text = r.payload.get("transcript", "")
                if current_chars + len(chunk_text) > MAX_CONTEXT_CHARS:
                    break # Stop packing to save context window
                    
                valid_chunks.append(r)
                current_chars += len(chunk_text)
                
            if not valid_chunks and final_points:
                # Fallback for meta-queries (e.g. "summarize") that have low semantic overlap
                valid_chunks = final_points[:10]
                for r in valid_chunks:
                    current_chars += len(r.payload.get("transcript", ""))
                
            # Build the compressed transcript
            video_transcripts = {}
            for r in valid_chunks:
                vid = r.payload.get("video_id")
                video_transcripts.setdefault(vid, []).append(r)
                
            full_transcript, file_summaries = format_transcript_data(
                video_transcripts, 
                prefix="[NOTE: The context was long. Providing dynamically packed high-relevance snippets.]"
            )
            
            # --- THOUGHT PROCESS: STEP 3 ---
            yield f"✅ *Packed {len(valid_chunks)} highly relevant segments ({current_chars // 4} tokens). Generating answer...*\n\n---\n\n"

        else:
             yield f"✅ *Video is short. Loaded full transcript. Generating answer...*\n\n---\n\n"

        # 2. Build the System Prompt
        is_folder_context = len(request.target_video_ids) > 1
        
        # --- GRAPH RAG INJECTION ---
        
        # Determine if it's a global query
        is_global_query = is_folder_context and any(word in request.query.lower() for word in ["summarize", "overall", "all", "themes", "connections"])
        
        if is_global_query:
            graph_context = graph_service.get_global_community_summaries(video_ids=request.target_video_ids)
        else:
            query_words = [w for w in request.query.split() if len(w) > 4]
            graph_context = graph_service.get_local_graph_context(query_words)
            if not graph_context.strip():
                graph_context = "No direct graph relationships found."

        context_desc = "The user is asking about the contents of a folder with multiple files." if is_folder_context else "The user is asking about a specific file."
        
        # Build a list of all files in the current context to prevent hallucination
        all_target_files = []
        for vid in request.target_video_ids:
            ext = media_lib.get(vid, "").split('.')[-1].lower()
            if ext == "pdf": ftype = "PDF"
            elif ext in ["jpg", "jpeg", "png", "webp"]: ftype = "Image"
            elif ext in ["mp3", "wav", "m4a"]: ftype = "Audio"
            elif ext in ["mp4", "mkv"]: ftype = "Video"
            else: ftype = "Media"
            all_target_files.append(f"- {vid} ({ftype})")
            
        folder_scope_str = "\n".join(all_target_files)

        system_instruction = f"""You are a helpful AI Assistant. You are chatting with the user about the following {len(request.target_video_ids)} file(s) in their current folder context:
{folder_scope_str}

{context_desc}

Based ONLY on the provided extracted data AND the Knowledge Graph Context, answer the user's question. 
If the user asks a general question about the folder (like how many files there are), use the file list provided above.
Keep your response conversational, concise, and direct. Do NOT provide a long detailed breakdown of the entire file unless the user explicitly asks for a summary.

Knowledge Graph Context:
{graph_context}

Relevant Extracted Data (May only contain snippets from a subset of the files):
{full_transcript}"""

        # 3. Format the chat history and stream the response
        provider = llm_settings.provider.lower()
        model_name = llm_settings.model

        stream_generator = llm_service.generate_chat_stream(
            system_instruction=system_instruction,
            chat_history=request.chat_history,
            query=request.query,
            provider=provider,
            model_name=model_name
        )
        
        for chunk in stream_generator:
            yield chunk

    return StreamingResponse(chat_flow(), media_type="text/event-stream")


# =====================================================================
# VIDEO MANAGEMENT: Delete a video from DB + disk
# =====================================================================
@router.delete("/video/{video_id}")
def delete_video(video_id: str, delete_file: bool = Query(default=True)):
    """
    Deletes all Qdrant vectors for a given video_id.
    If delete_file=True (default), also removes the media file from disk.
    """
    # 0. If currently ingesting, gracefully cancel it to drop file locks
    if video_id in active_ingestions and not active_ingestions[video_id]["done"]:
        active_ingestions[video_id]["canceled"] = True
        import time
        # Give the generator a moment to cleanly release the cv2 file lock
        time.sleep(1)

    qdrant_client = get_qdrant_client()
    
    # 1. Count points and delete from Qdrant if collection exists
    vectors_deleted = 0
    try:
        count_result = qdrant_client.count(
            collection_name="video_segments",
            count_filter=models.Filter(must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id)
                )
            ]),
            exact=True
        )
        vectors_deleted = count_result.count

        if vectors_deleted > 0:
            qdrant_client.delete(
                collection_name="video_segments",
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=video_id)
                        )
                    ])
                )
            )

        # 1.5 Delete from Knowledge Graph
        graph_service.delete_video(video_id)
    except Exception as e:
        print(f"Warning: Failed to delete qdrant or graph vectors: {e}")

    # 3. Optionally delete the media file from disk
    file_deleted = False
    if delete_file and os.path.exists("media"):
        for root, dirs, files in os.walk("media"):
            for filename in files:
                raw_name = os.path.splitext(filename)[0]
                candidate_id = re.sub(r'\W+', '_', raw_name.lower()).strip('_')
                if candidate_id == video_id:
                    os.remove(os.path.join(root, filename))
                    file_deleted = True

    return {
        "success": True,
        "video_id": video_id,
        "vectors_deleted": vectors_deleted,
        "file_deleted": file_deleted
    }

@router.post("/video/{video_id}/move")
def move_video(video_id: str, request: VideoMoveRequest):
    """Moves a video file to a new folder inside /media"""
    if not os.path.exists("media"):
        return {"success": False, "error": "Media directory not found."}
        
    # Sanitize new folder name
    clean_folder = re.sub(r'[^a-zA-Z0-9_\-\ /]', '', request.target_folder).strip()
    clean_folder = clean_folder.replace('//', '/').strip('/')
    target_dir = os.path.join("media", clean_folder) if clean_folder else "media"
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Find current file and any duplicates
    moved = False
    for root, dirs, files in os.walk("media"):
        for filename in files:
            raw_name = os.path.splitext(filename)[0]
            candidate_id = re.sub(r'\W+', '_', raw_name.lower()).strip('_')
            if candidate_id == video_id:
                old_path = os.path.join(root, filename)
                new_path = os.path.join(target_dir, filename)
                
                if old_path != new_path:
                    if os.path.exists(new_path):
                        # Destination already has a copy, deduplicate by deleting this extra copy
                        os.remove(old_path)
                    else:
                        shutil.move(old_path, new_path)
                moved = True
            
    return {"success": moved}


# =====================================================================
# SETTINGS: LLM Configuration
# =====================================================================
@router.get("/api/settings/llm")
def get_llm_settings():
    return llm_settings.model_dump()

@router.post("/api/settings/llm")
def update_llm_settings(new_settings: dict):
    try:
        import app.core.config as config
        updated = LLMSettings(**new_settings)
        # Update the module-level reference globally
        config.llm_settings = updated
        
        # Also update the local reference in routes.py
        global llm_settings
        llm_settings = updated
        
        save_settings(llm_settings)
        # Hot swap the clients
        llm_service.init_clients(llm_settings)
        return {"success": True, "message": "LLM Settings updated and loaded successfully."}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/graph")
def get_graph_data(target_video_ids: str = Query(None)):
    """Returns the knowledge graph nodes and edges for visualization."""
    if target_video_ids:
        # split comma separated string into list
        video_ids = [vid.strip() for vid in target_video_ids.split(",") if vid.strip()]
        return graph_service.export_graph_json(video_ids)
    return graph_service.export_graph_json()
