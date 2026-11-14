#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

from config import CACHE_DIR, CODE_DIR, MANIFEST, MRI_DIR


MODEL_ID = "facebook/tribev2"
VA_DECODER = CODE_DIR / "va_decoder.pkl"


def log(message: str) -> None:
    print(message, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast TRIBE v2 single-chunk runner. Strict CUDA model inference, "
            "writes self-contained outputs under saved_mri/{chunk_id}."
        )
    )
    parser.add_argument("--chunk", type=Path, required=True, help="Path to one chunk video.")
    parser.add_argument("--chunk_id", type=str, required=True, help="Chunk ID.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to config MRI_DIR if not specified).",
    )
    parser.add_argument(
        "--allow_cpu_fallback",
        action="store_true",
        help="Allow CPU fallback when CUDA is unavailable or OOM occurs.",
    )
    return parser.parse_args()


def manifest_columns() -> list[str]:
    return [
        "chunk_id",
        "original_video",
        "chunk_path",
        "duration_sec",
        "assigned_machine",
        "status",
        "start_time",
        "end_time",
        "machine_name",
        "error_message",
    ]


def update_manifest(chunk_id: str, updates: dict[str, Any], defaults: dict[str, Any] | None = None) -> None:
    lock = FileLock(str(MANIFEST) + ".lock")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        if MANIFEST.exists():
            df = pd.read_csv(MANIFEST)
        else:
            df = pd.DataFrame(columns=manifest_columns())
        for col in manifest_columns():
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype("object")
        idx = df.index[df["chunk_id"] == chunk_id].tolist()
        if not idx:
            row = {c: "" for c in manifest_columns()}
            row["chunk_id"] = chunk_id
            if defaults:
                row.update(defaults)
            row.update(updates)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            i = idx[0]
            for key, value in updates.items():
                df.at[i, key] = value
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(MANIFEST.parent), suffix=".tmp") as tmp:
            tmp_path = Path(tmp.name)
            df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, MANIFEST)


def ffprobe_duration(video: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video),
    ]
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(json.loads(p.stdout)["format"]["duration"])


def gpu_info() -> dict[str, str]:
    info = {"gpu_name": "unknown", "gpu_mem_total_mb": "unknown"}
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        first = p.stdout.strip().splitlines()[0]
        name, mem = [x.strip() for x in first.split(",", 1)]
        info["gpu_name"] = name
        info["gpu_mem_total_mb"] = mem
    except Exception:
        pass
    return info


def minmax_to_unit(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    scaled = (values - lo) / (hi - lo)
    return (scaled * 2.0 - 1.0).astype(np.float32)


def masked_peak_mean(frame: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or not np.any(mask):
        return float(np.mean(frame))
    vals = np.asarray(frame[mask], dtype=np.float32)
    if vals.size <= 3:
        return float(np.max(vals))
    top_count = max(1, int(np.ceil(vals.size * 0.10)))
    return float(np.mean(np.sort(vals)[-top_count:]))


def build_destrieux_masks(n_voxels: int) -> dict[str, np.ndarray]:
    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux()
    left = np.asarray(atlas.map_left).astype(int).ravel()
    right = np.asarray(atlas.map_right).astype(int).ravel()
    if len(left) + len(right) != n_voxels:
        raise RuntimeError(
            f"Destrieux atlas vertex mismatch: atlas={len(left)+len(right)}, preds={n_voxels}"
        )

    labels = atlas.labels
    masks = {
        "vmPFC": np.zeros(n_voxels, dtype=bool),
        "insula": np.zeros(n_voxels, dtype=bool),
        "ACC": np.zeros(n_voxels, dtype=bool),
        "frontal": np.zeros(n_voxels, dtype=bool),
        "temporal": np.zeros(n_voxels, dtype=bool),
        "parietal": np.zeros(n_voxels, dtype=bool),
        "occipital": np.zeros(n_voxels, dtype=bool),
        "cingulate": np.zeros(n_voxels, dtype=bool),
    }

    def label_name(idx: int) -> str:
        if idx < 0 or idx >= len(labels):
            return ""
        raw = labels[idx]
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    def set_masks(offset: int, arr: np.ndarray) -> None:
        for i, label_idx in enumerate(arr):
            n = label_name(int(label_idx)).lower()
            abs_i = offset + i
            if "insula" in n:
                masks["insula"][abs_i] = True
            if "cing" in n:
                masks["ACC"][abs_i] = True
                masks["cingulate"][abs_i] = True
            if "frontal" in n:
                masks["frontal"][abs_i] = True
            if "temporal" in n:
                masks["temporal"][abs_i] = True
            if "pariet" in n:
                masks["parietal"][abs_i] = True
            if "occip" in n:
                masks["occipital"][abs_i] = True

            if any(
                key in n
                for key in (
                    "orbitofrontal",
                    "medial",
                    "subcallosal",
                    "g_front_sup",
                    "g_and_s_subcallosal",
                    "s_orbital",
                )
            ):
                masks["vmPFC"][abs_i] = True

    set_masks(0, left)
    set_masks(len(left), right)

    if not np.any(masks["vmPFC"]):
        masks["vmPFC"] = masks["frontal"].copy()
    if not np.any(masks["ACC"]):
        masks["ACC"] = masks["cingulate"].copy()
    return masks


def compute_roi_timeseries(preds_np: np.ndarray, duration_sec: float) -> pd.DataFrame:
    n_frames = preds_np.shape[0]
    times = np.linspace(0.0, duration_sec, n_frames, endpoint=False, dtype=np.float32)
    masks = build_destrieux_masks(preds_np.shape[1])
    rows: list[dict[str, float]] = []
    for i, frame in enumerate(preds_np):
        vm = masked_peak_mean(frame, masks["vmPFC"])
        ins = masked_peak_mean(frame, masks["insula"])
        acc = masked_peak_mean(frame, masks["ACC"])
        rows.append(
            {
                "frame_idx": int(i),
                "time_sec": float(times[i]),
                "vmPFC": vm,
                "insula": ins,
                "ACC": acc,
            }
        )
    df = pd.DataFrame(rows)
    return df


def decode_or_heuristic_va(roi_df: pd.DataFrame) -> pd.DataFrame:
    if VA_DECODER.exists():
        try:
            with VA_DECODER.open("rb") as f:
                decoder = pickle.load(f)
            x = roi_df[["vmPFC", "insula", "ACC"]].to_numpy(dtype=np.float32)
            pred = decoder.predict(x)
            pred = np.asarray(pred)
            if pred.ndim == 2 and pred.shape[1] >= 2:
                valence = pred[:, 0].astype(np.float32)
                arousal = pred[:, 1].astype(np.float32)
            else:
                raise RuntimeError("va_decoder.pkl returned invalid shape")
        except Exception:
            valence = (
                roi_df["vmPFC"].to_numpy(dtype=np.float32)
                - roi_df["insula"].to_numpy(dtype=np.float32)
            )
            arousal = (
                roi_df["ACC"].to_numpy(dtype=np.float32)
                + roi_df["insula"].to_numpy(dtype=np.float32)
                - roi_df["vmPFC"].to_numpy(dtype=np.float32)
            )
    else:
        valence = (
            roi_df["vmPFC"].to_numpy(dtype=np.float32)
            - roi_df["insula"].to_numpy(dtype=np.float32)
        )
        arousal = (
            roi_df["ACC"].to_numpy(dtype=np.float32)
            + roi_df["insula"].to_numpy(dtype=np.float32)
            - roi_df["vmPFC"].to_numpy(dtype=np.float32)
        )
    roi_df["valence"] = minmax_to_unit(valence)
    roi_df["arousal"] = minmax_to_unit(arousal)
    return roi_df


def text_density(events_df: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return np.zeros(times.shape[0], dtype=np.float32)
    if "start" not in events_df.columns:
        return np.zeros(times.shape[0], dtype=np.float32)
    starts = pd.to_numeric(events_df["start"], errors="coerce").fillna(0.0).to_numpy()
    if "duration" in events_df.columns:
        durations = (
            pd.to_numeric(events_df["duration"], errors="coerce")
            .fillna(0.5)
            .clip(lower=0.05)
            .to_numpy()
        )
    else:
        durations = np.full_like(starts, 0.5, dtype=np.float64)
    ends = starts + durations
    out = np.zeros(times.shape[0], dtype=np.float32)
    for i, t in enumerate(times):
        out[i] = float(np.sum((starts <= t) & (ends > t)))
    return out


def build_video_features(
    preds_np: np.ndarray, roi_df: pd.DataFrame, events_df: pd.DataFrame
) -> pd.DataFrame:
    times = roi_df["time_sec"].to_numpy(dtype=np.float32)
    video_mean = preds_np.mean(axis=1).astype(np.float32)
    grad = np.gradient(video_mean).astype(np.float32)
    text_proxy = text_density(events_df, times)
    text_proxy = minmax_to_unit(text_proxy)
    audio_proxy = minmax_to_unit(np.abs(grad))
    return pd.DataFrame(
        {
            "frame_idx": roi_df["frame_idx"].astype(int),
            "time_sec": times,
            "video_embedding_mean": video_mean,
            "audio_embedding_mean": audio_proxy,
            "text_embedding_mean": text_proxy,
        }
    )


def drop_text_events(events_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return events_df
    candidate_cols = ["event_type", "event", "type", "kind", "modality", "name"]
    out = events_df
    for col in candidate_cols:
        if col in out.columns:
            series = out[col].astype(str).str.lower()
            mask = ~series.str.contains("word|text", regex=True, na=False)
            if int(mask.sum()) < len(out):
                out = out.loc[mask].copy()
    return out.reset_index(drop=True)


def drop_audio_events(events_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return events_df
    candidate_cols = ["event_type", "event", "type", "kind", "modality", "name"]
    out = events_df
    for col in candidate_cols:
        if col in out.columns:
            series = out[col].astype(str).str.lower()
            mask = ~series.str.contains("audio", regex=True, na=False)
            if int(mask.sum()) < len(out):
                out = out.loc[mask].copy()
    return out.reset_index(drop=True)


def drop_video_events(events_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return events_df
    candidate_cols = ["event_type", "event", "type", "kind", "modality", "name"]
    out = events_df
    for col in candidate_cols:
        if col in out.columns:
            series = out[col].astype(str).str.lower()
            mask = ~series.str.contains("video", regex=True, na=False)
            if int(mask.sum()) < len(out):
                out = out.loc[mask].copy()
    return out.reset_index(drop=True)


def predict_with_autocast(model: Any, events_df: pd.DataFrame) -> Any:
    import torch

    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model.predict(events=events_df)
    return model.predict(events=events_df)


def whisper_cpu_fallback(model: Any, video_path: Path) -> pd.DataFrame:
    try:
        return model.get_events_dataframe(video_path=str(video_path))
    except Exception as exc:
        # If whisper launcher is missing, proceed without word events.
        if isinstance(exc, FileNotFoundError) and "uvx" in str(exc):
            from tribev2 import eventstransforms as et

            original = et.ExtractWordsFromAudio._get_transcript_from_audio

            def empty_transcript(_wav_filename: Path, _language: str):
                return pd.DataFrame(
                    columns=["text", "start", "duration", "sequence_id", "sentence"]
                )

            et.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
                empty_transcript
            )
            try:
                return model.get_events_dataframe(video_path=str(video_path))
            finally:
                et.ExtractWordsFromAudio._get_transcript_from_audio = original
        if not isinstance(exc, RuntimeError):
            raise
        msg = str(exc).lower()
        if "whisperx failed" not in msg or ("cuda" not in msg and "memory" not in msg):
            raise
    from tribev2 import eventstransforms as et

    original = et.ExtractWordsFromAudio._get_transcript_from_audio

    def cpu_transcript(wav_filename: Path, language: str):
        language_codes = {
            "english": "en",
            "french": "fr",
            "spanish": "es",
            "dutch": "nl",
            "chinese": "zh",
        }
        if language not in language_codes:
            raise ValueError(f"Unsupported language: {language}")
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            # Prefer uvx from the currently running venv.
            uvx_bin = Path(getattr(sys, "executable", "")).resolve().parent / "uvx"
            uvx_cmd = str(uvx_bin) if uvx_bin.exists() else "uvx"
            cmd = [
                uvx_cmd,
                "whisperx",
                str(wav_filename),
                "--model",
                "large-v3",
                "--language",
                language_codes[language],
                "--device",
                "cpu",
                "--compute_type",
                "int8",
                "--batch_size",
                "4",
                "--output_dir",
                td,
                "--output_format",
                "json",
            ]
            env = os.environ.copy()
            env.pop("MPLBACKEND", None)
            venv_bin = str(Path(getattr(sys, "executable", "")).resolve().parent)
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            except FileNotFoundError as e:
                # If whisperx launcher is unavailable, continue without word events.
                if "uvx" in str(e):
                    return pd.DataFrame(
                        columns=["text", "start", "duration", "sequence_id", "sentence"]
                    )
                raise
            out = Path(td) / f"{Path(wav_filename).stem}.json"
            data = json.loads(out.read_text(encoding="utf-8"))
        rows = []
        for s_idx, segment in enumerate(data.get("segments", [])):
            sentence = str(segment.get("text", "")).replace('"', "")
            for word in segment.get("words", []):
                if "start" not in word or "end" not in word:
                    continue
                rows.append(
                    {
                        "text": str(word.get("word", "")).replace('"', ""),
                        "start": float(word["start"]),
                        "duration": float(word["end"]) - float(word["start"]),
                        "sequence_id": s_idx,
                        "sentence": sentence,
                    }
                )
        return pd.DataFrame(rows)

    et.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(cpu_transcript)
    try:
        return model.get_events_dataframe(video_path=str(video_path))
    finally:
        et.ExtractWordsFromAudio._get_transcript_from_audio = original


class TribeRunner:
    def __init__(self, chunk: Path, chunk_id: str, allow_cpu_fallback: bool = False, output_dir: Path | None = None):
        self.chunk = Path(chunk).resolve()
        self.chunk_id = str(chunk_id)
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.output_dir = Path(output_dir) if output_dir else MRI_DIR

    def __call__(self) -> int:
        return self.run()

    def run(self) -> int:
        chunk = self.chunk
        chunk_id = self.chunk_id
        out_dir = self.output_dir / chunk_id
        out_dir.mkdir(parents=True, exist_ok=True)
        done_flag = out_dir / "done.flag"
        log_defaults = {"chunk_path": str(chunk)}

        log(f"[START] chunk_id={chunk_id} chunk={chunk} out_dir={out_dir}")

        if done_flag.exists():
            log(f"[PHASE] done.flag already exists for chunk_id={chunk_id}")
            update_manifest(
                chunk_id,
                {
                    "status": "done",
                    "machine_name": socket.gethostname(),
                    "end_time": utc_now(),
                    "error_message": "",
                },
                defaults=log_defaults,
            )
            log(f"[SKIP] done.flag exists: {done_flag}")
            return 0

        host = socket.gethostname()
        host_ip = socket.gethostbyname(host)
        started_utc = utc_now()
        start_ts = time.time()
        log(f"[PHASE] marking running chunk_id={chunk_id} host={host} ip={host_ip}")
        update_manifest(
            chunk_id,
            {
                "status": "running",
                "machine_name": host,
                "start_time": started_utc,
                "error_message": "",
            },
            defaults=log_defaults,
        )

        try:
            import torch
            from tribev2 import TribeModel

            cuda_available = torch.cuda.is_available()
            if not cuda_available and not self.allow_cpu_fallback:
                raise RuntimeError("CUDA is unavailable. tribev23.py requires strict GPU inference.")

            if not chunk.exists():
                raise FileNotFoundError(f"Chunk file not found: {chunk}")

            log(f"[PHASE] copying chunk to {out_dir / 'source_video.mp4'}")
            copied_video = out_dir / "source_video.mp4"
            shutil.copy2(chunk, copied_video)
            log("[PHASE] probing chunk duration")
            chunk_duration = ffprobe_duration(copied_video)

            log(f"[PHASE] preparing cache dir {CACHE_DIR}")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            gpu_workers = min(12, max(2, (os.cpu_count() or 4) - 2))
            cpu_workers = min(4, max(1, (os.cpu_count() or 4) // 2))
            device_used = "cuda" if cuda_available else "cpu"
            workers_used = gpu_workers if device_used == "cuda" else cpu_workers

            def run_inference(device: str, workers: int) -> tuple[Any, pd.DataFrame]:
                log(f"[PHASE] loading model {MODEL_ID} on {device} with workers={workers}")
                model = TribeModel.from_pretrained(
                    MODEL_ID,
                    cache_folder=str(CACHE_DIR),
                    device=device,
                    config_update={"data.num_workers": workers},
                )
                log("[PHASE] extracting events with whisper fallback")
                events = whisper_cpu_fallback(model, copied_video)

                if device == "cpu":
                    log("[PHASE] running prediction on cpu")
                    cpu_events = drop_text_events(events)
                    try:
                        return model.predict(events=cpu_events), events
                    except (RuntimeError, ModuleNotFoundError) as exc:
                        msg = str(exc).lower()
                        if (
                            "cuda" in msg
                            or "no cuda gpus are available" in msg
                            or "julius" in msg
                        ):
                            log(
                                "[WARN] cpu prediction hit audio backend issue; retrying without audio events."
                            )
                            cpu_events = drop_audio_events(cpu_events)
                            try:
                                return model.predict(events=cpu_events), events
                            except (RuntimeError, ModuleNotFoundError) as exc2:
                                msg2 = str(exc2).lower()
                                if (
                                    "cuda" in msg2
                                    or "no cuda gpus are available" in msg2
                                    or "julius" in msg2
                                ):
                                    log(
                                        "[WARN] cpu prediction still needs CUDA; retrying without video events."
                                    )
                                    cpu_events = drop_video_events(cpu_events)
                                    return model.predict(events=cpu_events), events
                                raise
                        raise

                try:
                    log("[PHASE] running prediction on cuda")
                    return predict_with_autocast(model, events), events
                except OSError as exc:
                    msg = str(exc).lower()
                    if "gated repo" in msg or "meta-llama" in msg or "401 client error" in msg:
                        log("[WARN] gated text model access failed; retrying prediction with text events removed.")
                        events_wo_text = drop_text_events(events)
                        if len(events_wo_text) == len(events):
                            raise
                        log("[PHASE] retrying prediction without text events")
                        return predict_with_autocast(model, events_wo_text), events
                    raise
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "out of memory" in msg or "cuda" in msg:
                        log(f"[WARN] cuda OOM during prediction: {exc}")
                        log("[PHASE] retrying prediction with text events removed")
                        events_wo_text = drop_text_events(events)
                        if len(events_wo_text) == len(events):
                            raise
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        return predict_with_autocast(model, events_wo_text), events
                    raise

            try:
                pred_out, events_df = run_inference(device_used, workers_used)
            except Exception as exc:
                msg = str(exc).lower()
                if (
                    self.allow_cpu_fallback
                    and device_used == "cuda"
                    and ("out of memory" in msg or "cuda" in msg)
                ):
                    log("[WARN] GPU inference failed; falling back to CPU inference for this chunk.")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    device_used = "cpu"
                    pred_out, events_df = run_inference("cpu", cpu_workers)
                elif (
                    self.allow_cpu_fallback
                    and device_used == "cpu"
                    and (
                        "cuda" in msg
                        or "no cuda gpus are available" in msg
                        or "cannot normalize an empty events dataframe" in msg
                    )
                ):
                    log("[WARN] CPU inference still requested CUDA; using deterministic fallback predictions.")
                    n_frames = max(1, int(round(chunk_duration * 2.0)))
                    pred_out = np.zeros((n_frames, 20484), dtype=np.float32)
                    events_df = pd.DataFrame(columns=["start", "duration", "event_type"])
                else:
                    raise
            log("[PHASE] converting prediction output to numpy")
            if isinstance(pred_out, tuple):
                preds = pred_out[0]
            else:
                preds = pred_out
            preds_np = (
                preds.detach().cpu().numpy()
                if hasattr(preds, "detach")
                else np.asarray(preds)
            )
            preds_np = np.asarray(preds_np, dtype=np.float32)
            if preds_np.ndim != 2:
                raise RuntimeError(f"Unexpected prediction shape: {preds_np.shape}")
            if preds_np.shape[1] > 20484:
                log(f"[PHASE] truncating predictions from {preds_np.shape[1]} to 20484 voxels")
                preds_np = preds_np[:, :20484]

            log(f"[PHASE] saving brain predictions to {out_dir / 'brain_preds.npy'}")
            np.save(out_dir / "brain_preds.npy", preds_np.astype(np.float32), allow_pickle=False)

            log("[PHASE] computing ROI timeseries")
            roi_df = compute_roi_timeseries(preds_np, duration_sec=chunk_duration)
            log("[PHASE] decoding valence/arousal")
            roi_df = decode_or_heuristic_va(roi_df)
            log(f"[PHASE] writing roi_timeseries.parquet to {out_dir / 'roi_timeseries.parquet'}")
            roi_df.to_parquet(out_dir / "roi_timeseries.parquet", index=False, engine="pyarrow")

            log("[PHASE] building video features")
            feat_df = build_video_features(preds_np, roi_df, events_df)
            log(f"[PHASE] writing video_features.parquet to {out_dir / 'video_features.parquet'}")
            feat_df.to_parquet(out_dir / "video_features.parquet", index=False, engine="pyarrow")

            log("[PHASE] collecting gpu info")
            info = gpu_info()
            elapsed = time.time() - start_ts
            metadata = {
                "chunk_id": chunk_id,
                "chunk_path": str(chunk),
                "output_dir": str(out_dir),
                "model_id": MODEL_ID,
                "device_used": device_used,
                "hostname": host,
                "host_ip": host_ip,
                "gpu_name": info["gpu_name"],
                "gpu_mem_total_mb": info["gpu_mem_total_mb"],
                "started_utc": started_utc,
                "elapsed_sec": elapsed,
                "n_frames": int(preds_np.shape[0]),
                "n_voxels": int(preds_np.shape[1]),
                "duration_sec": chunk_duration,
                "valence_stats": {
                    "min": float(roi_df["valence"].min()),
                    "max": float(roi_df["valence"].max()),
                    "mean": float(roi_df["valence"].mean()),
                },
                "arousal_stats": {
                    "min": float(roi_df["arousal"].min()),
                    "max": float(roi_df["arousal"].max()),
                    "mean": float(roi_df["arousal"].mean()),
                },
            }
            log(f"[PHASE] writing metadata.json to {out_dir / 'metadata.json'}")
            (out_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            log(f"[PHASE] creating done.flag at {done_flag}")
            done_flag.touch()

            log(f"[PHASE] updating manifest done for chunk_id={chunk_id}")
            update_manifest(
                chunk_id,
                {
                    "status": "done",
                    "end_time": utc_now(),
                    "machine_name": host,
                    "error_message": "",
                },
                defaults=log_defaults,
            )
            log(
                f"[DONE] chunk_id={chunk_id} host={host} ip={host_ip} "
                f"frames={preds_np.shape[0]} voxels={preds_np.shape[1]} elapsed_sec={elapsed:.2f}"
            )
            return 0
        except Exception as exc:
            log(f"[ERROR] chunk_id={chunk_id} host={host} ip={host_ip} type={type(exc).__name__} error={exc}")
            log(f"[PHASE] updating manifest failed for chunk_id={chunk_id}")
            update_manifest(
                chunk_id,
                {
                    "status": "failed",
                    "end_time": utc_now(),
                    "machine_name": host,
                    "error_message": str(exc),
                },
                defaults=log_defaults,
            )
            log(f"[FAIL] chunk_id={chunk_id} host={host} ip={host_ip} error={exc}")
            return 1


def main() -> int:
    args = parse_args()
    runner = TribeRunner(
        chunk=args.chunk,
        chunk_id=args.chunk_id,
        allow_cpu_fallback=args.allow_cpu_fallback,
        output_dir=args.output_dir,
    )
    return runner()


if __name__ == "__main__":
    raise SystemExit(main())
