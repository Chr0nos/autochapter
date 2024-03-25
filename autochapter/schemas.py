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


class Chapter(BaseModel):
    index: int
    name: str
    start: timedelta

    def __str__(self) -> str:
        return f"CHAPTER{self.index:02}={self.time_format}\nCHAPTER{self.index:02}NAME={self.name}\n"

    @property
    def time_format(self) -> str:
        hours = self.start.seconds // 3600
        timeleft = timedelta(seconds=self.start.total_seconds() - hours * 3600)
        mins = int(timeleft.total_seconds() // 60)
        timeleft -= timedelta(minutes=mins)
        seconds = int(timeleft.total_seconds() // 1)
        miliseconds = int((timeleft.total_seconds() - seconds) * 1000)

        ms_txt = str(miliseconds)
        ms_len = len(ms_txt)
        ms_txt = ms_txt + ("0" * max(3 - ms_len, 0))
        return f"{hours:02}:{mins:02}:{seconds:02}.{ms_txt}"
