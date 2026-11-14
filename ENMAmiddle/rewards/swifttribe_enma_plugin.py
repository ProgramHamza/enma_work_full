from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch


# Root folder that holds the teacher models (expects models/tribev2, models/swift, models/swiftold).
# Set $ENMA_TEACHER_ROOT to point at it; defaults to ./teacher_models.
ENMA_ROOT = Path(os.environ.get("ENMA_TEACHER_ROOT", "teacher_models"))
ENMA_MODELS_ROOT = ENMA_ROOT / "models"
TRIBEV2_ROOT = ENMA_MODELS_ROOT / "tribev2"
SWIFT_ROOT = ENMA_MODELS_ROOT / "swift"
SWIFTOLD_ROOT = ENMA_MODELS_ROOT / "swiftold"

for path in (ENMA_ROOT, TRIBEV2_ROOT, SWIFT_ROOT, SWIFTOLD_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Defer imports to avoid ModuleNotFoundError when not used
# from tribev2.demo_utils import TribeModel
# from model_swift import create_swift_model
# from models.swiftold.model import fMRIEmotionTransformer


def _resolve_mount(path: Path) -> Path:
    if path.exists():
        return path
    text = str(path)
    if text.startswith("/mnt/tribe-share/"):
        alt = Path(text.replace("/mnt/tribe-share/", "/srv/tribe-share/", 1))
        if alt.exists():
            return alt
    if text.startswith("/srv/tribe-share/"):
        alt = Path(text.replace("/srv/tribe-share/", "/mnt/tribe-share/", 1))
        if alt.exists():
            return alt
    return path


def get_video_info(video_path: str | Path) -> tuple[float, int, float]:
    video_path = _resolve_mount(Path(video_path))
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0 and result.stdout.strip():
        try:
            info = json.loads(result.stdout)
            stream = info["streams"][0]
            fps_num, fps_den = stream["r_frame_rate"].split("/")
            fps = float(fps_num) / float(fps_den)
            num_frames = int(stream["nb_frames"])
            duration = num_frames / fps if fps > 0 else 0.0
            return fps, num_frames, duration
        except Exception:
            pass

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    duration = num_frames / fps if fps > 0 else 0.0
    return fps, num_frames, duration


def _filter_event_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "type" not in df.columns:
        return df
    return df[~df["type"].isin(["Word", "Text", "Sentence"])].copy()


def _load_affect_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    from model_swift import create_swift_model
    from models.swiftold.model import fMRIEmotionTransformer
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = {
        key.removeprefix("model.").removeprefix("module."): value
        for key, value in state_dict.items()
    }
 
    keys = set(state_dict.keys())
    if "input_projection.weight" in keys:
        input_dim = int(state_dict["input_projection.weight"].shape[1])
        model = fMRIEmotionTransformer(input_dim=input_dim).to(device)
        model.load_state_dict(state_dict, strict=True)
        return model
    if "backbone.surface_proj.weight" in keys:
        embed_dim = int(state_dict["backbone.surface_proj.weight"].shape[0])
        model = create_swift_model(
            in_channels=1,
            embed_dim=embed_dim,
            use_surface_input=True,
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        return model
    if "backbone.patch_embed.weight" in keys:
        embed_dim = int(state_dict["backbone.patch_embed.weight"].shape[0])
        model = create_swift_model(
            in_channels=1,
            embed_dim=embed_dim,
            use_surface_input=False,
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        return model
 
    raise RuntimeError(
        f"Could not detect a supported affect checkpoint type in {checkpoint_path}"
    )
 
 
def _load_tribe_model(device: str = "cpu") -> TribeModel:
    from tribev2.demo_utils import TribeModel
    local_tribev2 = TRIBEV2_ROOT
    config_update = {
        "data.audio_feature.device": device,
        "data.video_feature.image.device": device,
        "data.video_feature.image.batch_size": 64,
    }
    if local_tribev2.exists():
        try:
            return TribeModel.from_pretrained(local_tribev2, device=device, config_update=config_update)
        except Exception:
            pass
    return TribeModel.from_pretrained("facebook/tribev2", device=device, config_update=config_update)


def _frames_to_video(frames: Iterable[np.ndarray], fps: float, output_path: Path) -> None:
    frames = list(frames)
    if not frames:
        raise ValueError("No frames provided")
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] not in (3, 4):
        raise ValueError("Frames must be HxWx3 or HxWx4 arrays")
    height, width = int(first.shape[0]), int(first.shape[1])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video writer for {output_path}")
    try:
        for frame in frames:
            arr = np.asarray(frame)
            if arr.shape[0] != height or arr.shape[1] != width:
                raise ValueError("All frames must share the same resolution")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


@dataclass
class AffectPrediction:
    valence: float
    arousal: float
    window_index: int
    time_sec: float


class SwiftTribeENMAPlugin:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        window_seconds: float = 3600.0,
        skip_transcription: bool = False,
        max_windows: int | None = None,
        no_affect: bool = False,
    ) -> None:
        self.no_affect = no_affect
        self.window_seconds = float(window_seconds)
        self.skip_transcription = bool(skip_transcription)
        self.max_windows = max_windows
        self.device = torch.device(device)
        self.root_dir = ENMA_ROOT
        self._tribe_model: TribeModel | None = None
        self._affect_model: torch.nn.Module | None = None

        if self.no_affect:
            return

        if checkpoint_path is None:
            checkpoint_path = SWIFTOLD_ROOT / "checkpoints" / "best_model.pth"
        checkpoint_path = _resolve_mount(Path(checkpoint_path))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Affect checkpoint not found: {checkpoint_path}")

        # Skip loading the heavy tribe model to avoid import errors (only load affect model)
        self._affect_model = _load_affect_model(checkpoint_path, self.device)
        self._affect_model.eval()

    def _zero_scores(self, count: int = 1, condition: str = "baseline") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "condition": [condition] * count,
                "window_index": np.arange(count, dtype=np.int64),
                "time_sec": np.zeros(count, dtype=np.float32),
                "valence": np.zeros(count, dtype=np.float32),
                "arousal": np.zeros(count, dtype=np.float32),
                "valence_seq": [[]] * count,
                "arousal_seq": [[]] * count,
            }
        )

    def score_video(self, video_path: str | Path, condition: str = "baseline") -> pd.DataFrame:
        if self.no_affect:
            fps, num_frames, _ = get_video_info(video_path)
            window_size_frames = max(1, int(self.window_seconds * fps))
            num_windows = max(1, (num_frames + window_size_frames - 1) // window_size_frames)
            if self.max_windows is not None:
                num_windows = min(num_windows, self.max_windows)
            return self._zero_scores(num_windows, condition=condition)

        assert self._tribe_model is not None
        assert self._affect_model is not None

        video_path = _resolve_mount(Path(video_path))
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        fps, num_frames, _ = get_video_info(video_path)
        window_size_frames = max(1, int(self.window_seconds * fps))
        all_rows: list[dict[str, float | int | str | list[float]]] = []

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            for i, start_frame in enumerate(range(0, num_frames, window_size_frames)):
                if self.max_windows is not None and i >= self.max_windows:
                    break

                window_video = tmpdir_path / f"window_{i}.mp4"
                start_sec = start_frame / fps
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start_sec}",
                    "-t",
                    f"{self.window_seconds}",
                    "-i",
                    str(video_path),
                    "-r",
                    "10",
                    str(window_video),
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

                df = self._tribe_model.get_events_dataframe(
                    video_path=str(window_video),
                    audio_only=self.skip_transcription,
                )
                df_filtered = _filter_event_dataframe(df)
                brain_preds, _ = self._tribe_model.predict(events=df_filtered, verbose=False)
                if brain_preds.size == 0:
                    pred = np.zeros((1, 2), dtype=np.float32)
                else:
                    x = torch.from_numpy(brain_preds.astype(np.float32)).unsqueeze(1).to(self.device)
                    with torch.no_grad():
                        y = self._affect_model(x)
                    pred = np.atleast_2d(y.detach().cpu().numpy().squeeze())
                    if pred.ndim == 1:
                        pred = pred.reshape(1, -1)

                mean_valence = float(pred[:, 0].mean())
                mean_arousal = float(pred[:, 1].mean())
                time_sec = (start_frame / fps) + (self.window_seconds / 2.0)
                all_rows.append(
                    {
                        "condition": condition,
                        "window_index": i,
                        "time_sec": time_sec,
                        "valence": mean_valence,
                        "arousal": mean_arousal,
                        "valence_seq": pred[:, 0].tolist(),
                        "arousal_seq": pred[:, 1].tolist(),
                    }
                )

        return pd.DataFrame(all_rows)

    def score_frames(
        self,
        frames: Iterable[np.ndarray],
        fps: float = 30.0,
        condition: str = "baseline",
    ) -> pd.DataFrame:
        if self.no_affect:
            frames = list(frames)
            window_size_frames = max(1, int(self.window_seconds * fps))
            num_windows = max(1, (len(frames) + window_size_frames - 1) // window_size_frames)
            if self.max_windows is not None:
                num_windows = min(num_windows, self.max_windows)
            return self._zero_scores(num_windows, condition=condition)

        frames = list(frames)
        if not frames:
            return self._zero_scores(1)

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            video_path = tmpdir_path / "pov_capture.mp4"
            _frames_to_video(frames, fps=fps, output_path=video_path)
            return self.score_video(video_path, condition=condition)

    def score_one(self, video_path: str | Path, condition: str = "baseline") -> tuple[float, float]:
        df = self.score_video(video_path, condition=condition)
        if df.empty:
            return 0.0, 0.0
        return float(df["valence"].mean()), float(df["arousal"].mean())


def run_windowed_inference(
    video_path: str | Path,
    checkpoint_path: str | Path | None = None,
    window_duration: float = 3600.0,
    condition: str = "baseline",
    skip_transcription: bool = False,
    max_windows: int | None = None,
    no_affect: bool = False,
) -> pd.DataFrame:
    scorer = SwiftTribeENMAPlugin(
        checkpoint_path=checkpoint_path,
        window_seconds=window_duration,
        skip_transcription=skip_transcription,
        max_windows=max_windows,
        no_affect=no_affect,
    )
    return scorer.score_video(video_path, condition=condition)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(SWIFTOLD_ROOT / "checkpoints" / "best_model.pth"),
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--window", type=float, default=3600.0)
    parser.add_argument("--condition", type=str, default="baseline")
    parser.add_argument("--skip-transcription", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--no-affect", action="store_true")
    args = parser.parse_args()

    preds = run_windowed_inference(
        args.video,
        checkpoint_path=args.checkpoint,
        window_duration=args.window,
        condition=args.condition,
        skip_transcription=args.skip_transcription,
        max_windows=args.max_windows,
        no_affect=args.no_affect,
    )
    print(preds.to_string(index=False))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
