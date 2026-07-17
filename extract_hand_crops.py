#!/usr/bin/env python3
"""Extract 48x48 hand crops from labeled MP4 videos using an ONNX detector."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_MODEL = "deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query.onnx"
MODEL_SIZE = (640, 640)
LABELS = ("no_action", "point", "point_somewhere")
CROP_EXPANSION = 2.5


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass
class LabelStats:
    videos: int = 0
    frames: int = 0
    skipped_no_hand: int = 0
    skipped_too_many_hands: int = 0
    saved: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect hands in target MP4 files and save expanded 48x48 crops."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to ONNX model.")
    parser.add_argument("--input-dir", default=".", help="Directory containing MP4 files.")
    parser.add_argument("--output-dir", default=".", help="Directory where crops are written.")
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help=(
            "Specific MP4 file to process. May be specified multiple times. "
            "When set, --input-dir scanning is skipped."
        ),
    )
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--class-id", type=int, default=32)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--crop-size", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument(
        "--start-chunk-index",
        type=int,
        default=0,
        help="Chunk folder index for the first saved crop.",
    )
    parser.add_argument(
        "--start-image-index",
        type=int,
        default=0,
        help="Image filename index for the first saved crop.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into existing output label folders.",
    )
    return parser.parse_args()


def classify_video(path: Path) -> str | None:
    name = path.name
    if name.startswith("point_somewhere") and "_" in name:
        return "point_somewhere"
    if name.startswith("no_action") and "_" in name:
        return "no_action"
    if name.startswith("point") and not name.startswith("point_somewhere") and "_" in name:
        return "point"
    return None


def collect_videos(input_dir: Path) -> dict[str, list[Path]]:
    videos: dict[str, list[Path]] = {label: [] for label in LABELS}
    for path in sorted(input_dir.glob("*.mp4")):
        label = classify_video(path)
        if label is not None:
            videos[label].append(path)
    return videos


def collect_explicit_videos(video_args: list[str], input_dir: Path) -> dict[str, list[Path]]:
    videos: dict[str, list[Path]] = {label: [] for label in LABELS}
    for video_arg in video_args:
        path = Path(video_arg)
        if not path.exists() and not path.is_absolute():
            path = input_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {video_arg}")

        label = classify_video(path)
        if label is None:
            raise ValueError(f"Video filename does not match a target label pattern: {path}")
        videos[label].append(path)
    return videos


def create_session(model_path: Path) -> ort.InferenceSession:
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print(
            "WARNING: CUDAExecutionProvider is not available; ONNX Runtime will use CPU.",
            file=sys.stderr,
        )
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    try:
        session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception as exc:
        print(
            f"WARNING: Failed to create CUDA session ({exc}); falling back to CPU.",
            file=sys.stderr,
        )
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    providers = session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        print(
            f"WARNING: CUDAExecutionProvider is not active; active providers: {providers}",
            file=sys.stderr,
        )
    else:
        print(f"ONNX Runtime providers: {providers}")
    return session


def preprocess(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, MODEL_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]


def filter_hands(
    output: np.ndarray,
    frame_width: int,
    frame_height: int,
    class_id: int,
    score_threshold: float,
    min_size: int,
) -> list[Detection]:
    rows = output[0] if output.ndim == 3 else output
    detections: list[Detection] = []

    for row in rows:
        if row.shape[0] < 6:
            continue

        detected_class = int(round(float(row[0])))
        score = float(row[5])
        if detected_class != class_id or score < score_threshold:
            continue

        x1 = float(row[1]) * frame_width
        y1 = float(row[2]) * frame_height
        x2 = float(row[3]) * frame_width
        y2 = float(row[4]) * frame_height

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0.0, min(float(frame_width), x1))
        x2 = max(0.0, min(float(frame_width), x2))
        y1 = max(0.0, min(float(frame_height), y1))
        y2 = max(0.0, min(float(frame_height), y2))

        width = x2 - x1
        height = y2 - y1
        if width <= min_size or height <= min_size:
            continue

        detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score))

    return detections


def expanded_crop(frame: np.ndarray, detection: Detection, crop_size: int) -> np.ndarray | None:
    frame_height, frame_width = frame.shape[:2]
    box_width = detection.x2 - detection.x1
    box_height = detection.y2 - detection.y1
    center_x = (detection.x1 + detection.x2) / 2.0
    center_y = (detection.y1 + detection.y2) / 2.0
    expanded_width = box_width * CROP_EXPANSION
    expanded_height = box_height * CROP_EXPANSION

    x1 = int(np.floor(center_x - expanded_width / 2.0))
    y1 = int(np.floor(center_y - expanded_height / 2.0))
    x2 = int(np.ceil(center_x + expanded_width / 2.0))
    y2 = int(np.ceil(center_y + expanded_height / 2.0))

    x1 = max(0, min(frame_width, x1))
    x2 = max(0, min(frame_width, x2))
    y1 = max(0, min(frame_height, y1))
    y2 = max(0, min(frame_height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def output_path(
    output_dir: Path,
    label: str,
    image_index: int,
    chunk_size: int,
    start_chunk_index: int,
    start_image_index: int,
) -> Path:
    saved_offset = image_index - start_image_index
    chunk_index = start_chunk_index + (saved_offset // chunk_size)
    chunk_dir = output_dir / label / f"{chunk_index:06d}"
    return chunk_dir / f"{label}_{image_index:06d}.png"


def ensure_output_dirs(output_dir: Path, overwrite: bool, labels: Iterable[str]) -> None:
    labels = tuple(labels)
    existing = [output_dir / label for label in labels if (output_dir / label).exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output label folder already exists: {names}. "
            "Use --overwrite to allow writing into existing folders."
        )
    for label in labels:
        (output_dir / label).mkdir(parents=True, exist_ok=True)


def iter_frames(video_path: Path) -> Iterable[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_index, frame
            frame_index += 1
    finally:
        cap.release()


def process_video(
    video_path: Path,
    label: str,
    session: ort.InferenceSession,
    input_name: str,
    output_dir: Path,
    counters: dict[str, int],
    stats: LabelStats,
    args: argparse.Namespace,
) -> None:
    print(f"Processing {video_path.name} as {label}")
    stats.videos += 1

    for _, frame in iter_frames(video_path):
        stats.frames += 1
        frame_height, frame_width = frame.shape[:2]
        tensor = preprocess(frame)
        output = session.run(None, {input_name: tensor})[0]
        hands = filter_hands(
            output=output,
            frame_width=frame_width,
            frame_height=frame_height,
            class_id=args.class_id,
            score_threshold=args.score_threshold,
            min_size=args.min_size,
        )

        if not hands:
            stats.skipped_no_hand += 1
            continue
        if len(hands) >= 3:
            stats.skipped_too_many_hands += 1
            continue

        for hand in hands:
            crop = expanded_crop(frame, hand, args.crop_size)
            if crop is None:
                continue

            image_index = counters[label]
            path = output_path(
                output_dir,
                label,
                image_index,
                args.chunk_size,
                args.start_chunk_index,
                args.start_image_index,
            )
            if path.exists():
                raise FileExistsError(f"Output crop already exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), crop):
                raise RuntimeError(f"Failed to write crop: {path}")

            counters[label] += 1
            stats.saved += 1


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    model_path = Path(args.model)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if args.score_threshold < 0.0 or args.score_threshold > 1.0:
        raise ValueError("--score-threshold must be between 0.0 and 1.0")
    if args.min_size < 1:
        raise ValueError("--min-size must be at least 1")
    if args.crop_size < 1:
        raise ValueError("--crop-size must be at least 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    if args.start_chunk_index < 0:
        raise ValueError("--start-chunk-index must be at least 0")
    if args.start_image_index < 0:
        raise ValueError("--start-image-index must be at least 0")

    return model_path, input_dir, output_dir


def print_summary(stats_by_label: dict[str, LabelStats]) -> None:
    print("\nSummary")
    for label in LABELS:
        stats = stats_by_label[label]
        print(
            f"{label}: videos={stats.videos}, frames={stats.frames}, "
            f"skipped_no_hand={stats.skipped_no_hand}, "
            f"skipped_too_many_hands={stats.skipped_too_many_hands}, "
            f"saved={stats.saved}"
        )


def main() -> int:
    args = parse_args()
    try:
        model_path, input_dir, output_dir = validate_args(args)
        videos_by_label = (
            collect_explicit_videos(args.video, input_dir)
            if args.video
            else collect_videos(input_dir)
        )
        if not any(videos_by_label.values()):
            print("No target MP4 files found.", file=sys.stderr)
            return 1

        active_labels = [label for label, videos in videos_by_label.items() if videos]
        ensure_output_dirs(output_dir, args.overwrite, active_labels)
        session = create_session(model_path)
        input_name = session.get_inputs()[0].name
        counters = {label: args.start_image_index for label in LABELS}
        stats_by_label = {label: LabelStats() for label in LABELS}

        for label in LABELS:
            for video_path in videos_by_label[label]:
                process_video(
                    video_path=video_path,
                    label=label,
                    session=session,
                    input_name=input_name,
                    output_dir=output_dir,
                    counters=counters,
                    stats=stats_by_label[label],
                    args=args,
                )

        print_summary(stats_by_label)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
