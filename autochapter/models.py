from pydantic import BaseModel
from typing import Any
from datetime import timedelta


class Stream(BaseModel):
    index: int
    start_time: float
    codec_name: str
    codec_long_name: str
    codec_type: str
    tags: dict[str, Any]


class VideoStream(Stream):
    profile: str
    codec_tag_string: str
    codec_tag: str
    width: int
    height: int
    coded_width: int
    coded_height: int
    closed_captions: int
    pix_fmt: str
    level: int
    display_aspect_ratio: str


class ProbeStats(BaseModel):
    streams: list[VideoStream | Stream]


class FrameInfo(BaseModel):
    filename: str
    index: int
    offset: float
