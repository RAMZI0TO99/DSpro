from pydantic import BaseModel, Field
from typing import List, Dict

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=0)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    target_video_ids: List[str]
    folder_name: str = "Uncategorized"
    chat_history: List[ChatMessage]

class VideoMoveRequest(BaseModel):
    target_folder: str
