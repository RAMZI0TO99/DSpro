import os
import sys
import torch
import librosa
from qdrant_client import QdrantClient, models

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.vector_search import qdrant_client
from app.services.audio_extractor import AcousticExtractor

def find_media_file(video_id):
    """Recursively searches the media/ folder for the source file matching video_id."""
    for root, dirs, files in os.walk("media"):
        for f in files:
            if f == video_id or os.path.splitext(f)[0] == video_id:
                return os.path.join(root, f)
    return None

def main():
    collection_name = "video_segments"
    print(f"[*] Starting migration for Qdrant collection: '{collection_name}'")
    
    if not qdrant_client.collection_exists(collection_name=collection_name):
        print(f"[!] Collection '{collection_name}' does not exist. Nothing to migrate.")
        return

    # 1. Scroll and retrieve all points currently in the collection
    print(f"[*] Fetching points from Qdrant collection '{collection_name}'...")
    all_points = []
    offset = None
    while True:
        records, offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )
        all_points.extend(records)
        if offset is None:
            break
            
    print(f"[+] Snapshot retrieved. Found {len(all_points)} total points in collection.")
    if not all_points:
        print("[!] No points found. Terminating migration.")
        return

    # Initialize isolated extractor worker
    extractor = AcousticExtractor()

    # 3. Process points in memory and update the audio vector
    migrated_points = []
    print("[*] Re-embedding audio vectors using acoustic features...")
    
    for idx, point in enumerate(all_points):
        print(f"    [{idx+1}/{len(all_points)}] Processing point ID: {point.id}")
        payload = point.payload or {}
        video_id = payload.get("video_id")
        start_ts = payload.get("start_timestamp", 0.0)
        end_ts = payload.get("end_timestamp", 0.0)

        # Default vector dimension (512-d zeros)
        new_audio_vector = [0.0] * 512
        
        # Locate the media file
        media_file = find_media_file(video_id) if video_id else None
        
        if media_file:
            ext = media_file.lower().split('.')[-1]
            if ext in ["mp4", "avi", "mov", "mkv", "mp3", "wav", "m4a"]:
                extractor.load_audio_file(media_file)
                new_audio_vector = extractor.extract_slice_features(start_ts, end_ts)

        # Update point structure vectors
        orig_vectors = point.vector or {}
        visual_vector = orig_vectors.get("visual", [0.0]*768)
        text_sparse = orig_vectors.get("text_sparse", {"indices": [], "values": []})
        
        # Re-map text_sparse format into qdrant models.SparseVector
        if isinstance(text_sparse, dict):
            sparse_vector = models.SparseVector(
                indices=text_sparse.get("indices", []),
                values=text_sparse.get("values", [])
            )
        else:
            sparse_vector = text_sparse

        migrated_points.append(
            models.PointStruct(
                id=point.id,
                vector={
                    "visual": visual_vector,
                    "audio": new_audio_vector,
                    "text_sparse": sparse_vector
                },
                payload=payload
            )
        )

    # 4. Recreate the collection in Qdrant with new 512 dimensions for 'audio'
    print(f"[*] Re-creating Qdrant collection: '{collection_name}' with 512-d audio settings...")
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "visual": models.VectorParams(size=768, distance=models.Distance.COSINE),
            "audio": models.VectorParams(size=512, distance=models.Distance.COSINE),  # Upgraded to 512 dimensions
        },
        sparse_vectors_config={
            "text_sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
    )

    # 5. Re-upsert all migrated points
    print(f"[*] Re-inserting {len(migrated_points)} migrated points into '{collection_name}'...")
    for i in range(0, len(migrated_points), 50):
        qdrant_client.upsert(
            collection_name=collection_name,
            points=migrated_points[i:i+50]
        )
        
    print("[+] MIGRATION COMPLETE! All points re-embedded and upserted successfully.")

if __name__ == "__main__":
    main()
