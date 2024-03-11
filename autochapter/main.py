import ffmpeg
import click
import os
import io
import yaml

from annoy import AnnoyIndex
import faiss
import numpy as np
from PIL import Image
from pprint import pprint
from datetime import timedelta, time
from typing import Generator, Iterable
from pydantic import RootModel
from sentence_transformers import models, SentenceTransformer

from autochapter.models import ProbeStats, VideoStream, FrameInfo


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
    filename: str,
    vectors: Iterable[list[float]],
    fps: int,
) -> Generator[tuple[int, timedelta, list[float]], None, None]:
    frame_duraration = timedelta(seconds=1 / fps)
    current_frame_time = timedelta(seconds=0)
    for index, vector in enumerate(vectors):
        yield (index, current_frame_time, vector)
        current_frame_time += frame_duraration


@cli.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--fps", type=int, default=2)
def identify(folder: str, fps: int) -> None:
    # index = faiss.IndexFlatL2(512)
    index = AnnoyIndex(512, "angular")
    img_model = SentenceTransformer(modules=[models.CLIPModel()])

    current_index = 0
    metadata = {}

    for basename in sorted(os.listdir(folder)):
        filename = os.path.join(folder, basename)
        if basename.startswith(".") or not os.path.isfile(filename):
            continue
        vectors = generate_vectors(filename, img_model, fps)
        if not vectors:
            continue
        # put the vectors into the faiss index
        for frame_index, frame_time, frame_embedding in iter_over_vectors(filename, vectors, fps):
            index.add_item(current_index, frame_embedding)
            metadata[current_index] = FrameInfo(
                filename=filename,
                index=frame_index,
                offset=frame_time.total_seconds(),
            )
            current_index += 1
        # break

    print("Building index...")
    index.build(10)
    print("Saving index")
    index.save("/tmp/index.ann")

    with open("/tmp/index.yml", "w") as fp:
        fp.write(yaml.safe_dump({k: v.model_dump() for k, v in metadata.items()}))


@cli.command()
def search():
    # TODO: refactor all this shit
    # so far it's just for testing...
    with open("/tmp/index.yml") as fp:
        metadata = yaml.safe_load(fp)
    index = AnnoyIndex(512, "angular")
    index.load("/tmp/index.ann")

    print("Searching")

    for gid, frame in metadata.items():
        vectors, distances = index.get_nns_by_item(gid, 200, include_distances=True)

        relevant_vectors_ids = []
        for vector_id, distance in zip(vectors, distances):
            if distance > 0.2:
                continue
            # exclude vector from the same file
            if metadata[vector_id]["filename"] == metadata[gid]["filename"]:
                continue
            relevant_vectors_ids.append(vector_id)

        revelancy = len(relevant_vectors_ids)
        if revelancy > 20:
            print(
                metadata[gid]["filename"],
                revelancy,
                timedelta(seconds=frame["offset"]),
            )


if __name__ == "__main__":
    cli()
