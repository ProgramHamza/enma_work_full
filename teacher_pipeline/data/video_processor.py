#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterable

from filelock import FileLock

from config import CHUNKS_DIR, MACHINES, MANIFEST, MRI_DIR, VIDEOS_DIR


VIDEO_EXTS = {".mp4", ".webm", ".avi", ".mkv", ".mov"}


@dataclass
class ChunkItem:
    chunk_id: str
    original_video: Path
    chunk_path: Path
    duration_sec: float
    assigned_machine: str = ""
    status: str = "pending"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan videos, split into <=10 minute chunks, assign chunks to lab machines, "
            "and write the manifest.csv (location set in config.py)."
        )
    )
    parser.add_argument(
        "--max_chunk_duration",
        type=float,
        default=600.0,
        help="Maximum chunk duration in seconds (default: 600).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute and print actions without writing files or modifying manifest.",
    )
    return parser.parse_args()


def discover_videos(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(proc.stdout)
    return float(data["format"]["duration"])


def run_ffmpeg_segment_copy(
    input_video: Path, output_pattern: Path, segment_time: float, dry_run: bool
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-c",
        "copy",
        "-map",
        "0",
        "-f",
        "segment",
        "-segment_time",
        str(segment_time),
        "-reset_timestamps",
        "1",
        str(output_pattern),
    ]
    if dry_run:
        print("[DRY] ffmpeg:", " ".join(cmd))
        return
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "ffmpeg segment copy failed").strip()
        raise RuntimeError(f"ffmpeg segment copy failed for {input_video}: {detail}")


def safe_stem(path: Path) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in path.stem)


def ensure_unique_chunk_stem(base_stem: str, seen: set[str]) -> str:
    if base_stem not in seen:
        seen.add(base_stem)
        return base_stem
    i = 1
    while True:
        cand = f"{base_stem}_{i:02d}"
        if cand not in seen:
            seen.add(cand)
            return cand
        i += 1


def chunk_video(
    video_path: Path,
    max_chunk_duration: float,
    chunks_dir: Path,
    dry_run: bool,
    used_stems: set[str],
) -> list[ChunkItem]:
    total_duration = ffprobe_duration(video_path)
    stem = ensure_unique_chunk_stem(safe_stem(video_path), used_stems)
    chunk_items: list[ChunkItem] = []

    if total_duration <= max_chunk_duration:
        out_path = chunks_dir / f"{stem}_chunk_000.mp4"
        if dry_run:
            print(f"[DRY] link/copy {video_path} -> {out_path}")
        else:
            if out_path.exists() or out_path.is_symlink():
                out_path.unlink()
            try:
                out_path.symlink_to(video_path)
            except OSError:
                subprocess.run(
                    ["cp", "-f", str(video_path), str(out_path)],
                    check=True,
                )
        chunk_items.append(
            ChunkItem(
                chunk_id=out_path.stem,
                original_video=video_path,
                chunk_path=out_path,
                duration_sec=total_duration,
            )
        )
        return chunk_items

    temp_pattern = chunks_dir / f"{stem}_chunk_%03d.mp4"
    run_ffmpeg_segment_copy(video_path, temp_pattern, max_chunk_duration, dry_run=dry_run)

    if dry_run:
        num_chunks = int(total_duration // max_chunk_duration)
        if total_duration % max_chunk_duration > 0:
            num_chunks += 1
        for idx in range(num_chunks):
            out_path = chunks_dir / f"{stem}_chunk_{idx:03d}.mp4"
            dur = min(max_chunk_duration, total_duration - idx * max_chunk_duration)
            chunk_items.append(
                ChunkItem(
                    chunk_id=out_path.stem,
                    original_video=video_path,
                    chunk_path=out_path,
                    duration_sec=max(0.0, dur),
                )
            )
        return chunk_items

    produced = sorted(chunks_dir.glob(f"{stem}_chunk_*.mp4"))
    for out_path in produced:
        dur = ffprobe_duration(out_path)
        chunk_items.append(
            ChunkItem(
                chunk_id=out_path.stem,
                original_video=video_path,
                chunk_path=out_path,
                duration_sec=dur,
            )
        )
    return chunk_items


def weighted_round_robin(chunks: Iterable[ChunkItem]) -> list[ChunkItem]:
    weighted = []
    for ip, meta in MACHINES.items():
        weighted.extend([ip] * int(meta["weight"]))
    machine_cycle = cycle(weighted)
    out: list[ChunkItem] = []
    for item in chunks:
        item.assigned_machine = next(machine_cycle)
        out.append(item)
    return out


def write_manifest(chunks: list[ChunkItem], dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY] would write manifest: {MANIFEST}")
        return

    MRI_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "chunk_id": c.chunk_id,
            "original_video": str(c.original_video),
            "chunk_path": str(c.chunk_path),
            "duration_sec": f"{c.duration_sec:.3f}",
            "assigned_machine": c.assigned_machine,
            "status": c.status,
        }
        for c in chunks
    ]

    lock = FileLock(str(MANIFEST) + ".lock")
    with lock:
        with MANIFEST.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "chunk_id",
                    "original_video",
                    "chunk_path",
                    "duration_sec",
                    "assigned_machine",
                    "status",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)


def print_summary(video_count: int, chunks: list[ChunkItem]) -> None:
    print(f"[SUMMARY] videos_found={video_count}")
    print(f"[SUMMARY] chunks_total={len(chunks)}")
    by_machine = Counter(c.assigned_machine for c in chunks)
    for ip, count in sorted(by_machine.items()):
        name = MACHINES.get(ip, {}).get("name", ip)
        print(f"[ASSIGNMENT] {ip} ({name}) -> {count} chunks")


def main() -> int:
    args = parse_args()
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    videos = discover_videos(VIDEOS_DIR)
    print(f"[INFO] scanning: {VIDEOS_DIR}")
    print(f"[INFO] found videos: {len(videos)}")
    all_chunks: list[ChunkItem] = []
    used_stems: set[str] = set()

    for video in videos:
        try:
            chunks = chunk_video(
                video,
                max_chunk_duration=args.max_chunk_duration,
                chunks_dir=CHUNKS_DIR,
                dry_run=args.dry_run,
                used_stems=used_stems,
            )
            all_chunks.extend(chunks)
        except Exception as exc:
            print(f"[WARN] failed video {video}: {exc}")

    all_chunks.sort(key=lambda x: x.duration_sec, reverse=True)
    assigned = weighted_round_robin(all_chunks)
    write_manifest(assigned, dry_run=args.dry_run)
    print_summary(video_count=len(videos), chunks=assigned)
    print(f"[INFO] manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
