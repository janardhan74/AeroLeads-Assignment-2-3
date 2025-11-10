from __future__ import annotations
from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel, Field

from pydantic import BaseModel, Field
from typing import Optional

class BlogRequest(BaseModel):
    topic: str                          # keep topic required
    audience: Optional[str] = "Developers"
    tone: Optional[str] = "Clear, friendly"
    style: Optional[str] = "How-to tutorial"
    length: Optional[int] = 1400
    keywords: Optional[str] = None
    include_code: Optional[bool] = True
    include_tldr: Optional[bool] = True
    include_further_reading: Optional[bool] = True


class BlogMeta(BaseModel):
    title: str
    slug: str
    summary: str
    keywords: List[str] = Field(default_factory=list)
    reading_time_min: int = 8

class BlogState(TypedDict, total=False):
    request: BlogRequest
    outline_json: Dict[str, Any]
    draft_markdown: str
    refined_markdown: str
    meta: BlogMeta
    errors: List[str]
