from pydantic import BaseModel
from typing import List, Dict

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    target_video_ids: List[str]
    chat_history: List[ChatMessage]

class VideoMoveRequest(BaseModel):
    target_folder: str
