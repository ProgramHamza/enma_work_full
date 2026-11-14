# =====================================================================================================================
# NOTE: THIS IS NOT THE ORIGINAL TRAINING CODE.
#
# The original training script of the Caesar 3D ResNet student (`fast_cnn/train.py`) and its dataset class
# (`fast_cnn/dataset.py`) were left on the laboratory cluster in Japan and could not be recovered. This file is a
# re-implementation written afterwards so that the training procedure can be read and run. It has not been used to
# produce the Caesar weights or any number reported in the paper.
#
# Taken from the paper (Section III-F) and from the original files that survived
# (`test_fastcnn.py`, `snn_variant/caesar_eval.py`, `snn_variant/caesar_config.py`, `ENMAmiddle/training/train_enma.py`):
#   - model: torchvision r3d_18 initialised from Kinetics-400, last layer replaced by Linear(512 -> 2)
#   - input: 16 grey frames, 180 x 320, values in [-1, 1], repeated to 3 channels inside the model
#   - sampling: 16 frames with temporal stride 2, i.e. a 31-frame window of the source video
#   - data layout: one folder per video chunk with `source_video.mp4` and `swiFT_predictions.csv` (teacher labels)
#   - split 90 % / 10 % with seed 42, batch size 32, mixed precision, a CCC loss, flat-sample filter (0.15 / 0.5)
#   - checkpoint: `best_model.pth` holding the raw state_dict, the format `train_enma.py` and `test_fastcnn.py` load
#   - evaluation metrics: Pearson r, CCC, MSE, RMSE, MAE, mean and SD (paper Table II)
#
# Assumptions, because the original values are not known:
#   - optimiser (AdamW, lr 1e-4, weight decay 1e-4, cosine schedule), 30 epochs, early stopping after 15 epochs
#   - loss weights (2.0 x CCC + 0.05 x Huber, copied from the surviving trainer of the earlier spiking student)
#   - the split is made by chunk, the only augmentation is a horizontal flip, and the label file has the columns
#     `valence`, `arousal` and optionally `time_sec` (rows without a time are spread evenly over the chunk)
# =====================================================================================================================
"""Train and evaluate Caesar, the 3D ResNet-18 student that predicts valence and arousal from a 16-frame clip.

  python caesar/train_caesar.py --data_root data/saved_mri --out_dir checkpoints/caesar
  python caesar/train_caesar.py --data_root data/saved_mri --eval_only --ckpt checkpoints/caesar/best_model.pth
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

LOG = logging.getLogger("caesar")

IMAGE_SIZE = (180, 320)        # (height, width)
TEMPORAL_FRAMES = 16
TEMPORAL_STRIDE = 2
WINDOW = (TEMPORAL_FRAMES - 1) * TEMPORAL_STRIDE + 1   # 31 source frames
TRAIN_SPLIT = 0.9
SPLIT_SEED = 42
SHUFFLE_SEED = 1337
LABEL_FILES = ("swiFT_predictions.csv", "va_predictions.csv")
VIDEO_FILE = "source_video.mp4"
FLAT_THRESHOLD = 0.15          # |valence| and |arousal| below this = "flat" (near-neutral) sample
FLAT_KEEP_PROB = 0.5
CCC_WEIGHT, HUBER_WEIGHT = 2.0, 0.05


@dataclass(frozen=True)
class Sample:
    video: Path
    chunk: str
    start_frame: int
    valence: float
    arousal: float


class FastAffectCNN(nn.Module):
    """Caesar. Same definition as in `test_fastcnn.py` and `train_enma.py`, so the checkpoints are interchangeable."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models.video import R3D_18_Weights, r3d_18

        self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: (B, T, H, W) in [-1, 1]  ->  (B, 2) = [valence, arousal]."""
        return self.backbone(clip.unsqueeze(1).repeat(1, 3, 1, 1, 1))


def read_labels(path: Path, n_frames: int, fps: float) -> list[tuple[int, float, float]]:
    """Return (centre frame, valence, arousal) for every row of a teacher label file."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("valence") not in (None, "") and r.get("arousal") not in (None, "")]
    labels = []
    for index, row in enumerate(rows):
        if row.get("time_sec") not in (None, ""):
            centre = int(round(float(row["time_sec"]) * fps))
        else:  # no time column: spread the rows evenly over the chunk
            centre = int((index + 0.5) * n_frames / len(rows))
        labels.append((centre, float(row["valence"]), float(row["arousal"])))
    return labels


def chunk_samples(chunk_dir: Path) -> list[Sample]:
    video = chunk_dir / VIDEO_FILE
    label_path = next((chunk_dir / name for name in LABEL_FILES if (chunk_dir / name).exists()), None)
    if not video.exists() or label_path is None:
        return []
    capture = cv2.VideoCapture(str(video))
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    capture.release()
    if n_frames < WINDOW:
        return []
    samples = []
    for centre, valence, arousal in read_labels(label_path, n_frames, fps):
        start = min(max(centre - WINDOW // 2, 0), n_frames - WINDOW)
        samples.append(Sample(video, chunk_dir.name, start, valence, arousal))
    return samples


def split_chunks(data_root: Path) -> tuple[list[Path], list[Path]]:
    """90 % / 10 % split by chunk, so no validation chunk is ever seen in training."""
    chunks = sorted(p.parent for p in data_root.rglob(VIDEO_FILE))
    if len(chunks) < 2:
        raise SystemExit(f"need at least two chunk folders with {VIDEO_FILE} under {data_root}, found {len(chunks)}")
    random.Random(SPLIT_SEED).shuffle(chunks)
    n_train = min(max(int(round(len(chunks) * TRAIN_SPLIT)), 1), len(chunks) - 1)
    return chunks[:n_train], chunks[n_train:]


class VideoFrameDataset(Dataset):
    """Yields (uint8 clip of shape (16, 180, 320), float label [valence, arousal])."""

    def __init__(self, chunks: list[Path], train: bool, flat_filter: bool = True) -> None:
        samples = [s for chunk in chunks for s in chunk_samples(chunk)]
        if train and flat_filter:  # near-neutral samples are over-represented: keep half of them
            rng = random.Random(SHUFFLE_SEED)
            samples = [s for s in samples
                       if max(abs(s.valence), abs(s.arousal)) >= FLAT_THRESHOLD or rng.random() < FLAT_KEEP_PROB]
        self.samples = samples
        self.augment = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[index]
        capture = cv2.VideoCapture(str(sample.video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample.start_frame)
        frames = []
        for offset in range(WINDOW):
            ok, frame = capture.read()
            if not ok:
                break
            if offset % TEMPORAL_STRIDE == 0:
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(cv2.resize(grey, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA))
        capture.release()
        if not frames:
            raise RuntimeError(f"could not read frames from {sample.video} at {sample.start_frame}")
        while len(frames) < TEMPORAL_FRAMES:  # video ended early: repeat the last frame
            frames.append(frames[-1])
        clip = np.stack(frames)
        if self.augment and random.random() < 0.5:
            clip = clip[:, :, ::-1].copy()
        label = torch.tensor([sample.valence, sample.arousal], dtype=torch.float32)
        return torch.from_numpy(clip), label, index


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - concordance correlation coefficient (0 = perfect agreement)."""
    if target.var(unbiased=False) < 1e-6:
        return nn.functional.mse_loss(pred, target)
    pred_mean, target_mean = pred.mean(), target.mean()
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    denominator = pred.var(unbiased=False) + target.var(unbiased=False) + (pred_mean - target_mean) ** 2 + 1e-8
    return 1.0 - 2.0 * covariance / denominator


def total_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ccc = ccc_loss(pred[:, 0], target[:, 0]) + ccc_loss(pred[:, 1], target[:, 1])
    huber = nn.functional.smooth_l1_loss(pred, target, beta=0.1)
    return CCC_WEIGHT * ccc + HUBER_WEIGHT * huber


def to_model_input(clips: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 0..255  ->  float in [-1, 1], the normalisation the RL code applies to the camera frames."""
    return clips.to(device, non_blocking=True).float() / 255.0 * 2.0 - 1.0


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """The rows of the paper's Table II for one output (valence or arousal)."""
    error = pred - target
    covariance = float(np.mean((pred - pred.mean()) * (target - target.mean())))
    spread = float(pred.std() * target.std())
    return {
        "pearson_r": covariance / spread if spread > 0 else 0.0,
        "ccc": 2.0 * covariance / float(pred.var() + target.var() + (pred.mean() - target.mean()) ** 2 + 1e-8),
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "pred_mean": float(pred.mean()), "pred_sd": float(pred.std()),
        "gt_mean": float(target.mean()), "gt_sd": float(target.std()),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, dict[str, dict[str, float]]]:
    """Metrics per clip and per chunk (clip predictions averaged inside each chunk, as in the paper's N = 96)."""
    model.eval()
    preds, targets, indices = [], [], []
    for clips, labels, index in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            preds.append(model(to_model_input(clips, device)).float().cpu())
        targets.append(labels)
        indices.append(index)
    pred, target = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    chunk_names = [loader.dataset.samples[i].chunk for i in torch.cat(indices).tolist()]
    chunk_ids = np.array([sorted(set(chunk_names)).index(name) for name in chunk_names])
    by_chunk_pred = np.stack([pred[chunk_ids == c].mean(axis=0) for c in np.unique(chunk_ids)])
    by_chunk_target = np.stack([target[chunk_ids == c].mean(axis=0) for c in np.unique(chunk_ids)])
    result = {}
    for level, (p, t) in (("per_clip", (pred, target)), ("per_chunk", (by_chunk_pred, by_chunk_target))):
        result[level] = {"valence": metrics(p[:, 0], t[:, 0]), "arousal": metrics(p[:, 1], t[:, 1])}
    return result


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, device: torch.device) -> float:
    model.train()
    losses = []
    for clips, labels, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss = total_loss(model(to_model_input(clips, device)).float(), labels.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle and len(dataset) > args.batch_size)


def train(args: argparse.Namespace, device: torch.device) -> None:
    train_chunks, val_chunks = split_chunks(args.data_root)
    train_set = VideoFrameDataset(train_chunks, train=True, flat_filter=not args.no_flat_filter)
    val_set = VideoFrameDataset(val_chunks, train=False)
    LOG.info("chunks: %d train / %d validation; clips: %d / %d", len(train_chunks), len(val_chunks), len(train_set), len(val_set))
    train_loader, val_loader = make_loader(train_set, args, shuffle=True), make_loader(val_set, args, shuffle=False)
    model = FastAffectCNN(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_ccc, stale, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        val = evaluate(model, val_loader, device)["per_clip"]
        mean_ccc = 0.5 * (val["valence"]["ccc"] + val["arousal"]["ccc"])
        history.append({"epoch": epoch, "train_loss": loss, "val": val})
        LOG.info("epoch %d  loss %.4f  val CCC %.3f / %.3f  r %.3f / %.3f", epoch, loss, val["valence"]["ccc"],
                 val["arousal"]["ccc"], val["valence"]["pearson_r"], val["arousal"]["pearson_r"])
        if mean_ccc > best_ccc:
            best_ccc, stale = mean_ccc, 0
            torch.save(model.state_dict(), args.out_dir / "best_model.pth")  # raw state_dict, as the RL code expects
        else:
            stale += 1
            if stale >= args.patience:
                LOG.info("no improvement for %d epochs, stopping", stale)
                break
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    LOG.info("best mean validation CCC %.3f, weights in %s", best_ccc, args.out_dir / "best_model.pth")


def evaluate_checkpoint(args: argparse.Namespace, device: torch.device) -> None:
    _, val_chunks = split_chunks(args.data_root)
    loader = make_loader(VideoFrameDataset(val_chunks, train=False), args, shuffle=False)
    model = FastAffectCNN(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    report = evaluate(model, loader, device)
    LOG.info("held-out chunks: %d, clips: %d", len(val_chunks), len(loader.dataset))
    LOG.info("%s", json.dumps(report, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate the Caesar 3D ResNet student (re-implementation).")
    parser.add_argument("--data_root", type=Path, required=True, help="folder with one sub-folder per video chunk")
    parser.add_argument("--out_dir", type=Path, default=Path("checkpoints/caesar"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--no_pretrained", action="store_true", help="do not load the Kinetics-400 weights")
    parser.add_argument("--no_flat_filter", action="store_true", help="keep all near-neutral training samples")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/caesar/best_model.pth"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    torch.manual_seed(SHUFFLE_SEED)
    random.seed(SHUFFLE_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.eval_only:
        evaluate_checkpoint(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
