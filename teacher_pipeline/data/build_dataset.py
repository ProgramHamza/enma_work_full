#!/usr/bin/env python3
"""Assemble Tribe video + MRI emotion artifacts into a single dataset tree.

This script does not invent volumetric MRI data. It packages the actual artifacts
produced by the TRIBE video pipeline:

- source video/audio files
- brain_preds.npy
- roi_timeseries.parquet
- video_features.parquet
- metadata.json
- valence/arousal outputs when present

The result is a portable dataset layout under:
    <output-root>/<video_id>/<chunk_id>/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


def _swap_mount(path: Path) -> Path:
    text = str(path)
    if text.startswith("/srv/tribe-share/"):
        return Path(text.replace("/srv/tribe-share/", "/mnt/tribe-share/", 1))
    if text.startswith("/mnt/tribe-share/"):
        return Path(text.replace("/mnt/tribe-share/", "/srv/tribe-share/", 1))
    if text.startswith("/srv/tribeshare/"):
        return Path(text.replace("/srv/tribeshare/", "/mnt/tribeshare/", 1))
    if text.startswith("/mnt/tribeshare/"):
        return Path(text.replace("/mnt/tribeshare/", "/srv/tribeshare/", 1))
    return path


def _resolve_existing(path: Path) -> Path:
    if path.exists():
        return path
    alt = _swap_mount(path)
    if alt != path and alt.exists():
        return alt
    return path


def _iter_chunk_dirs(saved_mri_dir: Path) -> Iterable[Path]:
    for p in sorted(saved_mri_dir.glob("Youtube_*_chunk_*")):
        if p.is_dir():
            yield p


def _parse_chunk_id(chunk_dir: Path) -> str:
    return chunk_dir.name


def _parse_video_id(chunk_id: str) -> str:
    m = re.match(r"(Youtube_.+)_chunk_\d+$", chunk_id)
    return m.group(1) if m else "unknown_video"


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _select_media_files(chunk_dir: Path) -> list[Path]:
    wanted = [
        chunk_dir / "source_video.mp4",
        chunk_dir / "source_audio.wav",
        chunk_dir / "source_video.tsv",
        chunk_dir / "metadata.json",
        chunk_dir / "brain_preds.npy",
        chunk_dir / "roi_timeseries.parquet",
        chunk_dir / "video_features.parquet",
        chunk_dir / "va_predictions.csv",
        chunk_dir / "va_predictions.npy",
        chunk_dir / "va_predictions_meta.json",
    ]
    return [p for p in wanted if p.exists()]


@dataclass
class ItemRecord:
    video_id: str
    chunk_id: str
    source_chunk_dir: str
    dest_dir: str
    files: list[str]
    n_frames: int | None = None
    n_features: int | None = None
    has_valence_arousal: bool = False


def _read_shape_from_npy(path: Path) -> tuple[int, int] | None:
    try:
        import numpy as np

        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])
    except Exception:
        return None
    return None


def _has_va(parquet_path: Path) -> bool:
    if not parquet_path.exists():
        return False
    try:
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=None)
        return "valence" in df.columns and "arousal" in df.columns
    except Exception:
        return False


def build_dataset(
    saved_mri_dir: Path,
    output_root: Path,
    limit: int = 0,
    copy_existing_predictions: bool = True,
) -> list[ItemRecord]:
    saved_mri_dir = _resolve_existing(saved_mri_dir)
    output_root = _resolve_existing(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[ItemRecord] = []
    chunk_dirs = list(_iter_chunk_dirs(saved_mri_dir))
    if limit > 0:
        chunk_dirs = chunk_dirs[:limit]

    for chunk_dir in chunk_dirs:
        chunk_id = _parse_chunk_id(chunk_dir)
        video_id = _parse_video_id(chunk_id)
        dest_dir = output_root / video_id / chunk_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        copied: list[str] = []
        for src in _select_media_files(chunk_dir):
            if src.name in {"va_predictions.csv", "va_predictions.npy", "va_predictions_meta.json"} and not copy_existing_predictions:
                continue
            if _copy_if_exists(src, dest_dir / src.name):
                copied.append(src.name)

        n_frames = None
        n_features = None
        bp = chunk_dir / "brain_preds.npy"
        if bp.exists():
            shape = _read_shape_from_npy(bp)
            if shape is not None:
                n_frames, n_features = shape

        has_va = _has_va(chunk_dir / "roi_timeseries.parquet")
        records.append(
            ItemRecord(
                video_id=video_id,
                chunk_id=chunk_id,
                source_chunk_dir=str(chunk_dir),
                dest_dir=str(dest_dir),
                files=copied,
                n_frames=n_frames,
                n_features=n_features,
                has_valence_arousal=has_va or any(name.startswith("va_predictions") for name in copied),
            )
        )

        with open(dest_dir / "dataset_item.json", "w", encoding="utf-8") as f:
            json.dump(asdict(records[-1]), f, indent=2)

    manifest_path = output_root / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "chunk_id",
                "source_chunk_dir",
                "dest_dir",
                "files",
                "n_frames",
                "n_features",
                "has_valence_arousal",
            ],
        )
        writer.writeheader()
        for rec in records:
            row = asdict(rec)
            row["files"] = ";".join(rec.files)
            writer.writerow(row)

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a video_mri_emotion_dataset from TRIBE outputs")
    parser.add_argument(
        "--saved-mri-dir",
        default="data/saved_mri",
        help="Source saved_mri directory produced by tribev23",
    )
    parser.add_argument(
        "--output-root",
        default="data/video_mri_emotion_dataset",
        help="Destination dataset root",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of chunks to package (0 = all)",
    )
    parser.add_argument(
        "--no-copy-existing-predictions",
        action="store_true",
        help="Do not copy va_predictions.* files if they already exist in chunk folders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_dataset(
        saved_mri_dir=Path(args.saved_mri_dir),
        output_root=Path(args.output_root),
        limit=args.limit,
        copy_existing_predictions=not args.no_copy_existing_predictions,
    )
    print(f"Packaged {len(records)} chunks into {args.output_root}")
    if records:
        first = records[0]
        print(f"Example: {first.chunk_id} -> {first.dest_dir}")


if __name__ == "__main__":
    main()
