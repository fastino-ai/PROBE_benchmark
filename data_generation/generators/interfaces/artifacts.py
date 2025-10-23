"""
Shared artifact payload interfaces for emails, calendar events, and documents.

These payload shapes are used by both True Positives (retrieval targets) and
Distractors (confusers), so they live under generators/interfaces.
"""

from __future__ import annotations

from typing import List, Literal, Union
from pydantic import BaseModel, Field


class EmailPayload(BaseModel):
    subject: str
    sender: str
    to: List[str]
    cc: List[str] = Field(default_factory=list)
    timestamp: str  # ISO8601
    body: str


class CalendarPayload(BaseModel):
    title: str
    start_time: str  # ISO8601
    end_time: str  # ISO8601
    location: str
    attendees: List[str]
    description: str = ""


class DocumentPayload(BaseModel):
    title: str
    mime: Literal["text/markdown", "text/csv"]
    content: str


PayloadType = Union[EmailPayload, CalendarPayload, DocumentPayload]


__all__ = [
    "EmailPayload",
    "CalendarPayload",
    "DocumentPayload",
    "PayloadType",
]
