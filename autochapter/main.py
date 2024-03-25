import ffmpeg
import click
import os
import yaml
from sys import exit

from annoy import AnnoyIndex
import numpy as np
from PIL import Image
from datetime import timedelta, datetime
from typing import Generator, Iterable, Any
from sentence_transformers import models, SentenceTransformer
from functools import wraps
from autochapter.schemas import ProbeStats, VideoStream, FrameInfo, Chapter
from autochapter.models import File, Frame
from autochapter.config import cfg
import asyncio
from tortoise import Tortoise, connection
# from tortoise_vector.expression import CosineSimilarity
from asyncio_pool import AioPool
from tortoise.queryset import QuerySet

from tortoise.expressions import RawSQL


class CosineSimilarity(RawSQL):
    def __init__(self, field: str, vector: list[float], vector_size: int = 1536):
        super().__init__(f"{field} <-> '{vector}'")


MODEL_MAX_SIZE: int = 224
TORTOISE_CONFIG = {
    "connections": {
        "default": {
            "engine": "tortoise.backends.asyncpg",
            "credentials": {
                "database": cfg.postgres.db,
                "host": cfg.postgres.host,
                "port": cfg.postgres.port,
                "user": cfg.postgres.username,
                "password": cfg.postgres.password,
            }
        },
    },
    "apps": {
        "main": {
            "models": [
                "aerich.models",
                "autochapter.models",
            ],
            "default_connection": "default",
        },
    },
    "timezone": "UTC",
}


def sync_to_async(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


def with_database(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            await Tortoise.init(config=TORTOISE_CONFIG)
            return await func(*args, **kwargs)
        finally:
            await Tortoise.close_connections()

    return wrapper


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
    gpu: bool = False,
    quiet: bool = True,
) -> list[list[float]] | None:
    probe: dict[str, Any] = ffmpeg.probe(filename)
    # stream = ffmpeg.input(filename)
    # print(stream)
    stats: ProbeStats = ProbeStats.model_validate(probe)
    if stats.streams[0].codec_type == "video":
        video_stream_info: VideoStream = stats.streams[0]
        w, h = get_scaled_size(video_stream_info.width, video_stream_info.height, MODEL_MAX_SIZE)
        gpu_options = (
            {
                "hwaccel_device": "/dev/dri/renderD128",
                "hwaccel": "nvdec",
                "hwaccel_output_format": "nvenc",
            }
            if gpu
            else {}
        )
        raw_video, log = (
            ffmpeg.input(filename, **gpu_options)
            # .filter("fps", fps=fps)
            .filter("scale", w, h)
            # .filter('scale_npp', out_w=w, out_h=h)
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                r=fps,
            ).run(
                capture_stdout=True,
                quiet=quiet,
            )
        )
        frames: np.ndarray = np.frombuffer(raw_video, np.uint8).reshape([-1, h, w, 3])
        images = [Image.fromarray(frame) for frame in frames]
        vectors = img_model.encode(images)
        return [vector.tolist() for vector in vectors]


def iter_over_vectors(
    vectors: Iterable[list[float]],
    fps: int,
) -> Generator[tuple[timedelta, list[float]], None, None]:
    frame_duraration = timedelta(seconds=1 / fps)
    current_frame_time = timedelta(seconds=0)
    for vector in vectors:
        yield (current_frame_time, vector)
        current_frame_time += frame_duraration


@cli.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--fps", type=int, default=2, help="Frames per seconds to use into the model")
@click.option("--gpu", type=bool, is_flag=True, help="Enable GPU for video decode (nvdec)")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose FFMPEG output")
@sync_to_async
@with_database
async def build_index(folder: str, fps: int, gpu: bool, verbose: bool) -> None:
    """Build an index at the given folder"""
    img_model = SentenceTransformer(modules=[models.CLIPModel()])

    for basename in sorted(os.listdir(folder)):
        filename = os.path.join(folder, basename)
        if basename.startswith(".") or not os.path.isfile(filename):
            continue
        print(f"Vectorizing {datetime.utcnow()} {filename} ...")
        file = await File.create(filename=filename, fps=fps)
        vectors = generate_vectors(filename, img_model, fps, gpu, verbose is False)
        if not vectors:
            continue

        for frame_time, frame_embedding in iter_over_vectors(vectors, fps):
            await Frame.create(file=file, offset=frame_time, embedding=frame_embedding)


@cli.command()
@sync_to_async
@with_database
async def list_index() -> None:
    """List what files are currently in the index"""
    async for file in File.all():
        frames_count = await file.frames.all().count()
        print(f'File: {file.filename}: {frames_count} frame(s). (fps: {file.fps})')
    print("Total frames: ", await Frame.all().count())


@cli.command()
@click.option(
    "--min-group-size",
    type=int,
    default=60,
    help="How many frames must be in the group",
)
@click.option(
    "--max-delta",
    type=float,
    default=10.0,
    help="How much time can separate two frames",
)
@click.option(
    "--occurency-min",
    type=int,
    default=15,
    help="How many times a frame has to have neightboors",
)
@click.option(
    "--distance-max",
    type=float,
    default=0.26,
    help="Maxiumum distance between a frame and it's neigtboors",
)
@click.option(
    '--file',
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
    required=False,
    default=None
)
@sync_to_async
@with_database
async def search(
    min_group_size: int,
    max_delta: float,
    occurency_min: int,
    distance_max: float,
    file: str | None = None,
):
    """Search for chapters into indexed files"""

    print('Building index')
    folder = None
    index = await build_annoy_index(folder)
    print('Searching...')
    files_qs = File.all()
    if folder:
        files_qs = files_qs.filter(filename__startswith=folder)
    if file:
        files_qs = files_qs.filter(filename=file)
    async for file in files_qs:
        print(f'{file}:')
        relevant_frames = await get_relevant_frames(file, index, distance_max, occurency_min)
        groups = group_frames(relevant_frames, max_delta, min_group_size)
        # chapters = groups_to_chapters(groups)
        show_groups(groups)
        # show_chapters(chapters)


async def get_relevant_frames(
    file: File,
    index: AnnoyIndex,
    distance_max: float,
    occurences_min: int,
) -> list[Frame]:
    relevant_frames: list[Frame] = []
    current_file_frames_ids: list[int] = await file.frames.all().values_list('id', flat=True)
    # iterate over all frames from the current file.
    async for frame in file.frames.all().only('id', 'offset', 'file_id'):
        vectors, distances = index.get_nns_by_item(frame.id, 200, include_distances=True)
        valid_neightboors: list[int] = [
            vector
            for vector, distance in zip(vectors, distances)
            if distance <= distance_max and vector not in current_file_frames_ids
        ]
        neightboors_count: int = len(valid_neightboors)
        if neightboors_count >= occurences_min:
            relevant_frames.append(frame)

    return relevant_frames


async def get_relevant_frames_pg(file: File, distance_max: float, occurences_min: int) -> list[Frame]:
    relevant_frames: list[Frame] = []
    folder: str = str(os.path.dirname(file.filename))
    async for frame in file.frames.all():
        valid_neightboors_count_qs = (
            Frame
            .all()
            .exclude(file_id=file.id)
            .annotate(distance=CosineSimilarity("embedding", frame.embedding, 512))
            .filter(distance__lte=distance_max)
            .count()
        )
        print(await connection.connections.get('default').execute_query(f"EXPLAIN ANALYZE {valid_neightboors_count_qs.as_query()}"))
        valid_neightboors_count: int = await valid_neightboors_count_qs
        print(frame.offset, valid_neightboors_count)
        if valid_neightboors_count >= occurences_min:
            relevant_frames.append(frame)
    return relevant_frames



async def build_annoy_index(folder: str | None = None) -> AnnoyIndex:
    index = AnnoyIndex(512, 'angular')
    frames = Frame.all()
    if folder:
        frames = frames.filter(file__filename__startswith=folder)
    async for frame in frames.only("id", "embedding"):
        index.add_item(frame.id, frame.embedding)
    index.build(10)
    return index


def group_frames(
    frames: list[Frame],
    max_delta: float,
    min_group_size: int
) -> list[list[Frame]]:
    last_frame: Frame | None = None
    groups: list[list[Frame]] = []
    for frame in frames:
        # we create a new group if there is no new frame
        if not last_frame:
            groups.append([frame])
            last_frame = frame
        else:
            delta: timedelta = frame.offset - last_frame.offset
            # append the frame to the last group if still in the good delta
            if delta <= timedelta(seconds=max_delta):
                groups[-1].append(frame)
                last_frame = frame
            # otherwise we create a new group with the frame in it.
            else:
                groups.append([frame])
                last_frame = None
    return list(filter(lambda group: len(group) >= min_group_size, groups))


def groups_to_chapters(groups: list[list[Frame]]) -> list[Chapter]:
    """
    - https://blog.programster.org/add-chapters-to-mkv-file
    """
    if not groups:
        return []
    chapters: list[Chapter] = []
    i = 1

    # opening is right at file's start
    if groups[0][0].offset.total_seconds() > 0:
        chapters.append(Chapter(name="Intro", start=timedelta(seconds=0), index=i))
        i += 1

    chapters.append(
        Chapter(
            name="Opening",
            start=groups[0][0].offset,
            index=i,
        )
    )
    i += 1

    chapters.append(Chapter(name="Episode", start=groups[0][-1].offset, index=i))
    i += 1

    if len(groups) > 1:
        chapters.append(Chapter(name="Ending", start=groups[-1][0].offset ,index=i))

    return chapters


def show_chapters(chapters: Iterable[Chapter]) -> None:
    print("".join([str(chapter) for chapter in chapters]))


def show_groups(groups: list[list[Frame]]) -> None:
    for group_id, group in enumerate(groups):
        print(f'Group {group_id}: {group[0].offset} - {group[-1].offset} ({len(group)})')


if __name__ == "__main__":
    cli()
