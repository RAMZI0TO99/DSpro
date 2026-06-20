import os
import re
import shutil
from typing import List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import open_clip
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models


# --- 1. Load Environment Variables from .env ---
from openai import OpenAI

# --- . Initialize Local LLM Client (LM Studio) ---
local_client = OpenAI(
    base_url="http://127.0.0.1:1234/v1", 
    api_key="lm-studio"
)
LOCAL_MODEL = "gemma-4-e2b-it-Q4_K_M"

# =====================================================================
# THE SYSTEM PROMPT LIVES HERE (Global Scope)
# =====================================================================
SYSTEM_PROMPT = """
You are a highly precise Search Query Optimizer for a Tri-Modal Vector Database.
Your ONLY job is to take the user's raw input and convert it into a perfect, English-only keyword search string.

RULES:
1. Translate: If the user speaks in Arabic (or any other language), translate the core concepts to English.
2. Resolve Pronouns: If the user says "how does IT work", figure out what "it" is from context.
3. Strip Fluff: Remove conversational words like "please find", "show me", "extract the moment".
4. OUTPUT STRICTLY THE SEARCH STRING. DO NOT output any conversational text, quotes, or formatting.
"""

# --- Upload limits ---
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB
ALLOWED_MIME_TYPES = {"video/mp4", "video/x-matroska", "video/mkv"}

app = FastAPI(title="Agentic Spatio-Temporal RAG API")

# This mounts your media folder so videos can be streamed
app.mount("/media", StaticFiles(directory="media"), name="media")

# --- 2. Initialize Qdrant & AI Models ---
# When running in Docker, QDRANT_URL is injected by docker-compose (http://qdrant:6333).
# When running locally as a plain Python script, it falls back to the file-based DB.
_qdrant_url = os.getenv("QDRANT_URL")
if _qdrant_url:
    print(f"Connecting to Qdrant service at {_qdrant_url}")
    client = QdrantClient(url=_qdrant_url)
else:
    print("Falling back to local Qdrant file-based database at ./qdrant_db")
    client = QdrantClient(path="./qdrant_db")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading models on {device}...")

# Capture the 3rd output (image transform) for visual ingestion
clip_model, _, clip_transform = open_clip.create_model_and_transforms(
    'ViT-B-16-SigLIP', pretrained='webli'
)
clip_model = clip_model.to(device).eval()
clip_tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP-256')

print("Loading BM25 Lexical Engine...")
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")


# --- 3. Pydantic Models ---
class SearchQuery(BaseModel):
    query: str
    top_k: int = 1

class ChatQuery(BaseModel):
    query: str
    video_id: str
    chat_history: List[Dict[str, str]] = []


# --- 4. Initialize Hot Ingestion Pipeline (shares CLIP + DB client) ---
from ingestion import LocalHotIngestionPipeline

ingestion_engine = LocalHotIngestionPipeline(
    clip_model=clip_model,
    clip_tokenizer=clip_tokenizer,
    clip_transform=clip_transform,
    db_client=client
)


# =====================================================================
# HELPERS
# =====================================================================
def optimize_query_with_llm(raw_query: str) -> str:
    """Uses Local LM Studio (Gemma) to translate and optimize the raw user query."""
    try:
        response = local_client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_query}
            ],
            temperature=0.1, # Keep it low so Gemma doesn't hallucinate fluff
        )
        optimized_string = response.choices[0].message.content.strip()
        print(f"[Local LLM Router] Raw: '{raw_query}' -> Optimized: '{optimized_string}'")
        return optimized_string
    except Exception as e:
        print(f"[Local LLM Error] Is LM Studio running? Falling back to raw. Error: {e}")
        return raw_query

# =====================================================================
# ROUTES
# =====================================================================

@app.get("/thumbnail/{video_id}")
async def get_thumbnail(video_id: str):
    """Extracts a representative frame from a video and returns it as a JPEG thumbnail."""
    import cv2
    from fastapi.responses import Response

    if not os.path.exists("media"):
        raise HTTPException(status_code=404, detail="Media directory not found.")

    # Find the media file for this video_id
    video_file = None
    for filename in os.listdir("media"):
        raw_name = os.path.splitext(filename)[0]
        candidate_id = re.sub(r'\W+', '_', raw_name.lower()).strip('_')
        if candidate_id == video_id:
            video_file = os.path.join("media", filename)
            break

    if not video_file:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")

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
    new_h = int(h * (new_w / w))
    frame = cv2.resize(frame, (new_w, new_h))

    # Encode as JPEG in memory
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the Vanilla HTML/JS Frontend."""
    with open("Frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()


# --- UPLOAD ENDPOINT ---
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Saves an uploaded video to /media with MIME type and size validation.
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

    file_path = f"media/{file.filename}"
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
    raw_name = os.path.splitext(file.filename)[0]
    clean_id = re.sub(r'\W+', '_', raw_name.lower()).strip('_')

    return {
        "message": "Upload successful",
        "video_id": clean_id,
        "file_path": file_path,
        "size_bytes": bytes_written
    }


# --- INGEST STREAMING ENDPOINT ---
@app.get("/ingest/{video_id}")
async def ingest_video_stream(video_id: str, file_path: str):
    """Streams the progress of the ingestion engine via Server-Sent Events."""
    return StreamingResponse(
        ingestion_engine.process_video_stream(file_path, video_id),
        media_type="text/event-stream"
    )


# --- LIBRARY ENDPOINT ---
@app.get("/library")
def get_media_library():
    """Dynamically scans the media folder and returns all video IDs, titles, and file sizes."""
    video_lib = {}
    title_lib = {}
    size_lib = {}
    media_dir = "media"

    if not os.path.exists(media_dir):
        return {"videoLibrary": {}, "titleLibrary": {}, "sizeLibrary": {}}

    for filename in os.listdir(media_dir):
        if filename.endswith(".mp4") or filename.endswith(".mkv"):
            raw_name = os.path.splitext(filename)[0]
            clean_id = re.sub(r'\W+', '_', raw_name.lower()).strip('_')

            video_lib[clean_id] = f"/media/{filename}"
            title_lib[clean_id] = raw_name.replace("_", " ")
            size_lib[clean_id] = os.path.getsize(os.path.join(media_dir, filename))

    return {
        "videoLibrary": video_lib,
        "titleLibrary": title_lib,
        "sizeLibrary": size_lib
    }


# =====================================================================
# INTENT A: Search & Seek (Tri-Modal Vector Search)
# =====================================================================
@app.post("/search")
def search_video(request: SearchQuery):
    """Tri-modal hybrid search using CLIP visual + CLIP audio-text + BM25, fused with RRF."""
    # 1. Intercept and Optimize with LLM
    optimized_query = optimize_query_with_llm(request.query)

    # 2. Generate Dense Semantic Vector (CLIP)
    with torch.no_grad():
        text_input = clip_tokenizer([optimized_query]).to(device)
        dense_tensor = clip_model.encode_text(text_input)
        dense_tensor /= dense_tensor.norm(dim=-1, keepdim=True)
        dense_vector = dense_tensor.tolist()[0]

    # 3. Generate Sparse Lexical Vector (BM25)
    sparse_result = list(bm25_model.embed([optimized_query]))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist()
    )

    # 4. Tri-Modal Hybrid Search with RRF fusion
    search_results = client.query_points(
        collection_name="video_segments",
        prefetch=[
            models.Prefetch(query=dense_vector, using="visual", limit=20),
            models.Prefetch(query=dense_vector, using="audio", limit=20),
            models.Prefetch(query=sparse_vector, using="text_sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=request.top_k
    )

    output = []
    THRESHOLD = 0.04  # Confidence threshold to prevent hallucinations (raised from 0.02)

    for point in search_results.points:
        if point.score < THRESHOLD:
            continue

        transcript = point.payload.get("transcript", "")
        # Return up to 500 chars for the result detail card
        snippet = (transcript[:500] + "...") if len(transcript) > 500 else transcript

        output.append({
            "video_id": point.payload.get("video_id"),
            "start_timestamp": point.payload.get("start_timestamp"),
            "end_timestamp": point.payload.get("end_timestamp"),
            "matched_transcript": snippet,
            "hybrid_rrf_score": round(point.score, 4),
            "llm_optimized_query": optimized_query
        })

    if not output:
        return {"results": [], "message": "Out of domain query. No relevant video segments found."}

    return {"results": output}


# =====================================================================
# INTENT B: Chat & Summarize (Context-Aware LLM generation)
# =====================================================================
@app.post("/chat")
def chat_with_video(request: ChatQuery):
    """Handles multi-turn Q&A and summarization about a specific video."""

    # 1. Retrieve the full transcript for the target video
    records, _ = client.scroll(
        collection_name="video_segments",
        scroll_filter=models.Filter(must=[
            models.FieldCondition(
                key="video_id",
                match=models.MatchValue(value=request.video_id)
            )
        ]),
        limit=1000,
        with_payload=True,
        with_vectors=False
    )

    if not records:
        return {"answer": f"Error: No data found for video '{request.video_id}' in the database."}

    # ---> THIS IS THE LINE YOU WERE MISSING <---
    records.sort(key=lambda x: x.payload.get("start_timestamp", 0))
    full_transcript = " ".join([r.payload.get("transcript", "") for r in records])

    # --- HARDWARE LIMIT FIX ---
    # 1 token is roughly 4 characters. By limiting to 8000 chars (~2000 tokens), 
    # we force the GPU to process the prompt in ~6 seconds instead of 65 seconds.
    MAX_CHARS = 8000
    if len(full_transcript) > MAX_CHARS:
        full_transcript = full_transcript[:MAX_CHARS] + "\n...[TRANSCRIPT TRUNCATED DUE TO GPU MEMORY LIMITS]"

    # 2. Build the System Prompt with the Transcript
    system_instruction = f"""You are an AI Video Assistant. Based ONLY on the following video transcript, answer the user's question.
    Transcript:
    {full_transcript}"""
    # 3. Format the chat history for Gemma natively
    messages = [{"role": "system", "content": system_instruction}]
    
    if request.chat_history:
        for msg in request.chat_history:
            # Map frontend roles to OpenAI format
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
            
    # Add the current user query
    messages.append({"role": "user", "content": request.query})

    # 4. Generate the response locally
    try:
        response = local_client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=messages,
            temperature=0.7
        )
        return {"answer": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"answer": f"API Error: Make sure LM Studio local server is running! Details: {e}"}


# =====================================================================
# VIDEO MANAGEMENT: Delete a video from DB + disk
# =====================================================================
@app.delete("/video/{video_id}")
def delete_video(video_id: str, delete_file: bool = Query(default=True)):
    """
    Deletes all Qdrant vectors for a given video_id.
    If delete_file=True (default), also removes the media file from disk.
    """
    # 1. Count points before deletion for the response
    count_result = client.count(
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

    # 2. Delete from Qdrant
    client.delete(
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

    # 3. Optionally delete the media file from disk
    file_deleted = False
    if delete_file and os.path.exists("media"):
        for filename in os.listdir("media"):
            raw_name = os.path.splitext(filename)[0]
            candidate_id = re.sub(r'\W+', '_', raw_name.lower()).strip('_')
            if candidate_id == video_id:
                os.remove(os.path.join("media", filename))
                file_deleted = True
                break

    return {
        "success": True,
        "video_id": video_id,
        "vectors_deleted": vectors_deleted,
        "file_deleted": file_deleted
    }


if __name__ == "__main__":
    import uvicorn
    import asyncio
    import sys

    # Silences the WinError 10054 connection drops on Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(app, host="127.0.0.1", port=8000)