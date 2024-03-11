import ffmpeg
import click
import os
import io

import faiss
import numpy as np
from PIL import Image
from pprint import pprint
from datetime import timedelta, time
from typing import Generator, Iterable
from sentence_transformers import models, SentenceTransformer

from autochapter.models import ProbeStats, VideoStream


def get_scaled_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    aspect_ratio = target_width / width
    target_height = height * aspect_ratio
    return (int(target_width), int(target_height))


@click.group()
def cli(): ...


def generate_vectors(
    filename: str,
    img_model: SentenceTransformer,
    fps: int,
) -> list[list[float]] | None:
    probe: dict[str, Any] = ffmpeg.probe(filename)
    # stream = ffmpeg.input(filename)
    # print(stream)
    stats: ProbeStats = ProbeStats.model_validate(probe)
    if stats.streams[0].codec_type == "video":
        video_stream_info: VideoStream = stats.streams[0]
        w, h = get_scaled_size(video_stream_info.width, video_stream_info.height, 224)
        raw_video, log = (
            ffmpeg.input(filename)
            .filter("fps", fps=fps)
            .filter("scale", w, h)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(
                capture_stdout=True,
                # quiet=True,
            )
        )
        print(len(raw_video))
        frames: np.ndarray = np.frombuffer(raw_video, np.uint8).reshape([-1, h, w, 3])
        images = [Image.fromarray(frame) for frame in frames]
        vectors = img_model.encode(images)
        return [vector.tolist() for vector in vectors]


def iter_over_vectors(
    vectors: Iterable[list[float]],
    fps: int,
) -> Generator[tuple[timedelta, list[float]], None, None]:
    frame_duraration = 1 / fps
    current_frame_time = timedelta(seconds=0)
    for vector in vectors:
        yield (current_frame_time, vector)
        current_frame_time += timedelta(seconds=frame_duraration)


@cli.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--fps", type=int, default=2)
def identify(folder: str, fps: int) -> None:
    index = faiss.IndexFlatL2(512)
    img_model = SentenceTransformer(modules=[models.CLIPModel()])

    for basename in sorted(os.listdir(folder)):
        filename = os.path.join(folder, basename)
        if basename.startswith(".") or not os.path.isfile(filename):
            continue
        vectors = generate_vectors(filename, img_model, fps)
        # put the vectors into the faiss index
        for frame_time, frame_embedding in iter_over_vectors(vectors, fps):
            index.add(frame_embedding, {"frame_time": frame_time, "filename": filename})
        break

    faiss.write_index("/tmp/autochapter.index")


if __name__ == "__main__":
    cli()
