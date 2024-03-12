import ffmpeg
import click
import os
import yaml

from annoy import AnnoyIndex
import numpy as np
from PIL import Image
from datetime import timedelta, datetime
from typing import Generator, Iterable, Any
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
    gpu: bool = False,
    quiet: bool = True,
) -> list[list[float]] | None:
    probe: dict[str, Any] = ffmpeg.probe(filename)
    # stream = ffmpeg.input(filename)
    # print(stream)
    stats: ProbeStats = ProbeStats.model_validate(probe)
    if stats.streams[0].codec_type == "video":
        video_stream_info: VideoStream = stats.streams[0]
        w, h = get_scaled_size(video_stream_info.width, video_stream_info.height, 224)
        gpu_options = {
            'hwaccel_device': '/dev/dri/renderD128',
            "hwaccel": "nvdec",
            # 'hwaccel_output_format': 'cuda',
        } if gpu else {}
        raw_video, log = (
            ffmpeg.input(filename, **gpu_options)
            .filter("fps", fps=fps)
            .filter("scale", w, h)
            # .filter('scale_npp', out_w=w, out_h=h)
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                # r=fps,
            )
            .run(
                capture_stdout=True,
                quiet=quiet,
            )
        )
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
@click.option("--fps", type=int, default=2, help="Frames per seconds to use into the model")
@click.option("--gpu", type=bool, is_flag=True, help="Enable GPU for video decode (nvdec)")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose FFMPEG output")
def build_index(folder: str, fps: int, gpu: bool, verbose: bool) -> None:
    # index = faiss.IndexFlatL2(512)
    index = AnnoyIndex(512, "angular")
    img_model = SentenceTransformer(modules=[models.CLIPModel()])

    current_index = 0
    metadata = {}

    for basename in sorted(os.listdir(folder)):
        filename = os.path.join(folder, basename)
        if basename.startswith(".") or not os.path.isfile(filename):
            continue
        print(f"Vectorizing {datetime.utcnow()} {filename} ...")
        vectors = generate_vectors(filename, img_model, fps, gpu, verbose is False)
        if not vectors:
            continue
        # put the vectors into the index
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

    dump_metadata(metadata, "/tmp/index.yml")


def dump_metadata(metadata: dict[int, Any], filename: str) -> None:
    obj = {}
    for frame_id, frame_info in metadata.items():
        frame_filename = frame_info.filename
        if frame_filename not in obj:
            obj[frame_filename] = []
        obj[frame_filename].append(
            {
                "id": frame_id,
                "offset": frame_info.offset,
                "index": frame_info.index,
            }
        )
    with open(filename, "w") as fp:
        fp.write(yaml.safe_dump(obj))


def load_metadata(filename: str) -> dict[int, Any]:
    metadata = {}
    with open(filename, "r") as fp:
        obj = yaml.safe_load(fp.read())
    for filename, values in obj.items():
        for frame in values:
            metadata[frame["id"]] = {
                "filename": filename,
                "offset": frame["offset"],
                "index": frame["index"],
            }
    return metadata


@cli.command()
@click.option("--min-group-size", type=int, default=10)
@click.option("--max-delta", type=float, default=10.0)
@click.option("--occurency-min", type=int, default=20)
def search(
    min_group_size: int,
    max_delta: float,
    occurency_min: int,
):
    # TODO: refactor all this shit
    # so far it's just for testing...
    print("Loading metadata...")
    metadata = load_metadata("/tmp/index.yml")

    print("Loading index...")
    index = AnnoyIndex(512, "angular")
    index.load("/tmp/index.ann")

    print("Searching")

    files_mapping: dict[str, list[int]] = {}

    for frame_id, frame in metadata.items():
        filename = frame["filename"]
        offset = frame["offset"]
        if filename not in files_mapping:
            files_mapping[filename] = []

        vectors, distances = index.get_nns_by_item(frame_id, 200, include_distances=True)
        relevant_vectors_ids = []
        for vector_id, distance in zip(vectors, distances):
            if distance > 0.22:
                break
            # exclude vector from the same file
            if metadata[vector_id]["filename"] == filename:
                continue
            relevant_vectors_ids.append(vector_id)

        # how many time a similar frame happens in others files
        occurences = len(relevant_vectors_ids)
        if occurences >= occurency_min:
            metadata[frame_id]["n"] = occurences
            files_mapping[filename].append(frame_id)

    print("Making groups...")
    files_groups = {
        filename: group_frames_ids(files_mapping[filename], metadata, max_delta, min_group_size)
        for filename in files_mapping.keys()
    }
    show_files_mapping(files_groups, metadata)


def group_frames_ids(
    frames_ids: list[int],
    metadata: dict[int, Any],
    max_delta: float,
    min_group_size: int,
) -> list[list[int]]:
    """groups ids by their placement in time
    allowing a `max_delta` (in seconds) between frames
    (to handle missing frames from previous filtering)
    remove any group that is smaller than `min_group_size`
    """
    last_id: int | None = None
    last_frame_info = None
    groups: list[list[int]] = []

    for frame_id in frames_ids:
        frame_info = metadata[frame_id]
        if not last_id:
            groups.append([frame_id])
        elif not last_frame_info:
            groups.append([frame_id])
        else:
            delta = frame_info["offset"] - last_frame_info["offset"]
            if delta <= max_delta:
                groups[-1].append(frame_id)
            else:
                groups.append([frame_id])
                last_frame_info = None

        last_id = frame_id
        last_frame_info = frame_info

    return list(filter(lambda group: len(group) >= min_group_size, groups))


def show_files_mapping(files_groups: dict[str, list[list[int]]], metadata: dict[int, Any]) -> None:
    for filename, groups in files_groups.items():
        print(filename)
        for group_index, group in enumerate(groups):
            print(timedelta(seconds=metadata[group[-1]]["offset"] - metadata[group[0]]["offset"]))
            for frame_id in group:
                frame_offset = timedelta(seconds=metadata[frame_id]["offset"])
                n = metadata[frame_id]["n"]
                print(f"- {frame_id:3} [{n:2}] -> {frame_offset}")
            print("---")


if __name__ == "__main__":
    cli()
