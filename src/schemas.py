from typing import List, Literal, Optional, Union

from pydantic import BaseModel


class ImageURL(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


Content = Union[str, List[ContentPart]]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    timestamp: Optional[int] = None
    content: Content


class AddRequest(BaseModel):
    request_id: str
    messages: List[Message]
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: Content
    options: Optional[List[str]] = None
    user_id: str
    top_k: int