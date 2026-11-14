# =====================================================================================================================
# NOTE: THIS IS NOT THE ORIGINAL TRAINING CODE.
#
# The original training and evaluation scripts of the SwiFT stage of the teacher, and the model classes they used
# (`model_swift`, `models/swift`, `models/swiftold`), were left on the laboratory cluster in Japan and could not be
# recovered. This file is a re-implementation written afterwards so that the procedure can be read and run. It has
# not been used to produce the teacher weights, the pseudo-labels or any number reported in the paper, and its
# checkpoints are not guaranteed to load into the original model classes.
#
# Taken from the paper (Sections III-F and IV-A) and from the original files that survived
# (`swift_affect_regression.py`, `ENMAmiddle/rewards/swifttribe_enma_plugin.py`, `caesar/snn_variant/caesar_config.py`):
#   - task: regress continuous valence and arousal from the cortical activity that TRIBE v2 predicts for a video
#   - input: a sequence of predicted cortical maps, shape (time, 20484 vertices); the model is called as model(x)
#     with x of shape (batch, time, vertices) and returns (batch, 2) = [valence, arousal]
#   - model size: hidden width 256, 4 Transformer layers, 8 heads, dropout 0.2, sequence length 20
#   - training: 100 epochs, batch size 8, learning rate 1e-4, weight decay 1e-4, patience 80, best checkpoint
#     saved as `swift_best.pt` with the weights under the key `model_state_dict`
#   - evaluation on held-out clips with Pearson r, CCC, MSE, RMSE, MAE, mean and SD (paper Table I, N = 121)
#
# Assumptions, because the original values are not known:
#   - the exact layers (a linear projection of each cortical map, learned positions, a standard Transformer encoder,
#     mean pooling over time, a linear head), the AdamW optimiser and the mean-squared-error loss
#   - the data layout: one folder per clip with `brain_preds.npy` and a label file with the columns `valence` and
#     `arousal` (several rows are averaged to one label per clip)
#   - each clip is standardised with its own mean and SD, and the held-out set is a random 10 % (or `--val_clips`)
# =====================================================================================================================
"""Train and evaluate the SwiFT-style Transformer that maps predicted cortical activity to valence and arousal.

  python teacher_pipeline/swift/train_swift.py --data_root data/friday --out_dir checkpoints/swift
  python teacher_pipeline/swift/train_swift.py --data_root data/friday --eval_only --ckpt checkpoints/swift/swift_best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

LOG = logging.getLogger("swift")

INPUT_DIM = 20484      # cortical vertices predicted by TRIBE v2
HIDDEN_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.2
SEQ_LEN = 20
SPLIT_SEED = 42
BRAIN_FILE = "brain_preds.npy"
LABEL_FILES = ("va_labels.csv", "labels.csv")


class AffectTransformer(nn.Module):
    """Sequence of cortical maps (B, T, vertices) -> (B, 2) = [valence, arousal]."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS,
                 num_heads: int = NUM_HEADS, dropout: float = DROPOUT, seq_len: int = SEQ_LEN) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=4 * hidden_dim, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        maps = maps[:, : self.seq_len]
        tokens = self.input_projection(maps) + self.position[:, : maps.shape[1]]
        return self.head(self.norm(self.encoder(tokens).mean(dim=1)))


def read_label(folder: Path) -> tuple[float, float] | None:
    path = next((folder / name for name in LABEL_FILES if (folder / name).exists()), None)
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [(float(r["valence"]), float(r["arousal"])) for r in csv.DictReader(handle)
                if r.get("valence") not in (None, "") and r.get("arousal") not in (None, "")]
    if not rows:
        return None
    valence, arousal = np.mean(rows, axis=0)
    return float(valence), float(arousal)


def find_clips(data_root: Path) -> list[tuple[Path, tuple[float, float]]]:
    clips = []
    for brain in sorted(data_root.rglob(BRAIN_FILE)):
        label = read_label(brain.parent)
        if label is not None:
            clips.append((brain, label))
    if len(clips) < 2:
        raise SystemExit(f"need at least two clip folders with {BRAIN_FILE} and a label file under {data_root}")
    return clips


def split_clips(clips: list, val_fraction: float, val_clips: int | None) -> tuple[list, list]:
    """Random held-out set, fixed by a seed and never used in training."""
    shuffled = list(clips)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    n_val = val_clips if val_clips is not None else int(round(len(shuffled) * val_fraction))
    n_val = min(max(n_val, 1), len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


class CorticalClipDataset(Dataset):
    """Yields (float32 maps of shape (SEQ_LEN, vertices), float label [valence, arousal])."""

    def __init__(self, clips: list[tuple[Path, tuple[float, float]]], seq_len: int, train: bool) -> None:
        self.clips, self.seq_len, self.train = clips, seq_len, train

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.clips[index]
        maps = np.load(path).astype(np.float32)
        maps = maps.reshape(maps.shape[0], -1)
        maps = (maps - maps.mean()) / (maps.std() + 1e-6)
        if maps.shape[0] > self.seq_len:  # long clip: random window in training, centre window otherwise
            spare = maps.shape[0] - self.seq_len
            start = random.randint(0, spare) if self.train else spare // 2
            maps = maps[start:start + self.seq_len]
        elif maps.shape[0] < self.seq_len:  # short clip: repeat the last map
            maps = np.concatenate([maps, np.repeat(maps[-1:], self.seq_len - maps.shape[0], axis=0)])
        return torch.from_numpy(maps), torch.tensor(label, dtype=torch.float32)


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """The rows of the paper's Table I for one output (valence or arousal)."""
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, dict[str, float]]:
    model.eval()
    preds, targets = [], []
    for maps, labels in loader:
        preds.append(model(maps.to(device)).cpu())
        targets.append(labels)
    pred, target = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    combined_mse = float(np.mean((pred - target) ** 2))
    return {"valence": metrics(pred[:, 0], target[:, 0]), "arousal": metrics(pred[:, 1], target[:, 1]),
            "combined": {"mse": combined_mse, "rmse": float(np.sqrt(combined_mse)), "n": float(len(pred))}}


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
    model.train()
    losses = []
    for maps, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(maps.to(device)), labels.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def build_model(args: argparse.Namespace, device: torch.device) -> AffectTransformer:
    return AffectTransformer(input_dim=args.input_dim, seq_len=args.seq_len).to(device)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_clips, val_clips = split_clips(find_clips(args.data_root), args.val_fraction, args.val_clips)
    LOG.info("clips: %d train / %d held out", len(train_clips), len(val_clips))
    train_set = CorticalClipDataset(train_clips, args.seq_len, train=True)
    val_set = CorticalClipDataset(val_clips, args.seq_len, train=False)
    return (DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers),
            DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers))


def train(args: argparse.Namespace, device: torch.device) -> None:
    train_loader, val_loader = make_loaders(args)
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_mse, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": loss, "val": val})
        LOG.info("epoch %d  loss %.4f  val MSE %.4f  r %.3f / %.3f  CCC %.3f / %.3f", epoch, loss,
                 val["combined"]["mse"], val["valence"]["pearson_r"], val["arousal"]["pearson_r"],
                 val["valence"]["ccc"], val["arousal"]["ccc"])
        if val["combined"]["mse"] < best_mse:
            best_mse, stale = val["combined"]["mse"], 0
            torch.save({"model_state_dict": model.state_dict(), "input_dim": args.input_dim, "seq_len": args.seq_len,
                        "epoch": epoch}, args.out_dir / "swift_best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                LOG.info("no improvement for %d epochs, stopping", stale)
                break
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    LOG.info("best held-out MSE %.4f, weights in %s", best_mse, args.out_dir / "swift_best.pt")


def evaluate_checkpoint(args: argparse.Namespace, device: torch.device) -> None:
    _, val_loader = make_loaders(args)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
    args.input_dim, args.seq_len = int(checkpoint["input_dim"]), int(checkpoint["seq_len"])
    model = build_model(args, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    LOG.info("%s", json.dumps(evaluate(model, val_loader, device), indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate the SwiFT-style affect regressor (re-implementation).")
    parser.add_argument("--data_root", type=Path, required=True, help="folder with one sub-folder per clip")
    parser.add_argument("--out_dir", type=Path, default=Path("checkpoints/swift"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--input_dim", type=int, default=INPUT_DIM)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--val_clips", type=int, default=None, help="exact number of held-out clips (paper: 121)")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/swift/swift_best.pt"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    torch.manual_seed(SPLIT_SEED)
    random.seed(SPLIT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.eval_only:
        evaluate_checkpoint(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
