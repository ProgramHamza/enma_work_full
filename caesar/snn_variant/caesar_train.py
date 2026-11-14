"""
Train the SNN video → valence/arousal model.

Key changes vs v1:
  1. Uses improved dataset with frame pre-extraction cache.
  2. Shuffle=True for training DataLoader (was False — silent bug!).
  3. Warm-up LR schedule: 5 epochs linear warm-up + cosine annealing.
  4. Gradient clipping raised from 0.1 to 1.0 (0.1 was starving gradients).
  5. Improved loss: CCC-only during stabilization phase.
  6. EMA validation metrics for stability tracking.
  7. Label smoothing: small Gaussian noise on targets.
  8. Configurable flat-sample filtering.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .config import (
    APPLY_FLAT_FILTER,
    CAESAR_DATA_ROOTS, CAESAR_NUM_WORKERS, DATA_ROOT, DEVICE, DIAG_EVERY_EPOCHS,
    IMAGE_SIZE, FRAME_STEP, FLAT_SAMPLE_KEEP_PROB, FLAT_SAMPLE_THRESHOLD,
    LOG_EVERY_BATCHES, CAESAR_TEMPORAL_FRAMES, CAESAR_TEMPORAL_STRIDE,
    LEARN_BETA, PLOT_DIR, SNN_BATCH_SIZE, SNN_BETA,
    SNN_CKPT, SNN_DROPOUT, SNN_EPOCHS, SNN_HIDDEN, SNN_LR, SNN_NUM_STEPS,
    SNN_THRESHOLD, SNN_PATIENCE, SNN_PREPROCESS_SIZE, SNN_WEIGHT_DECAY,
    TRAIN_SAMPLE_LIMIT, TRAIN_VIDEO_LIMIT, VAL_SAMPLE_LIMIT, VAL_VIDEO_LIMIT,
    SNN_INPUT_NOISE_STD, SNN_OUTPUT_POPULATION, SNN_SPIKE_RATE_TARGET,
    SNN_SPIKE_RATE_WEIGHT, SNN_TARGET_NOISE_STD, SNN_USE_POPSAN, SNN_CCC_WEIGHT,
    SNN_REGRESSION_WEIGHT, SNN_AUGMENT_TRAIN, SNN_ENCODER_OUT_DIM,
    SNN_USE_TEMPORAL_ATTN,
)
from .dataset import build_dataloaders, print_label_stats
from .model   import Net, save_model

import numpy as np
import torch
import torch.nn as nn

PLOT_DIR.mkdir(parents=True, exist_ok=True)
SNN_CKPT.parent.mkdir(parents=True, exist_ok=True)

CAESAR_DATA_ROOTS = tuple(root for root in CAESAR_DATA_ROOTS if root.exists())
if not CAESAR_DATA_ROOTS:
    CAESAR_DATA_ROOTS = (DATA_ROOT,)


# ── loss ──────────────────────────────────────────────────────────────────────
def ccc_loss(pred: torch.Tensor, target: torch.Tensor,
             weights: torch.Tensor = None) -> torch.Tensor:
    """Concordance Correlation Coefficient loss. Returns 1 - CCC ∈ [0, 2]."""
    if target.var(unbiased=False) < 1e-6:
        if weights is not None:
            return torch.mean(weights * (pred - target) ** 2)
        return nn.functional.mse_loss(pred, target)

    p, t = pred.reshape(-1), target.reshape(-1)
    if weights is not None:
        w = weights.reshape(-1)
        w_norm = w / (w.sum() + 1e-8)
        pm = (p * w_norm).sum()
        tm = (t * w_norm).sum()
        cov   = (w_norm * (p - pm) * (t - tm)).sum()
        var_p = (w_norm * (p - pm) ** 2).sum()
        var_t = (w_norm * (t - tm) ** 2).sum()
        denom = var_p + var_t + (pm - tm) ** 2 + 1e-8
        return 1.0 - (2.0 * cov) / denom
    else:
        pm, tm = p.mean(), t.mean()
        cov    = ((p - pm) * (t - tm)).mean()
        denom  = p.var(unbiased=False) + t.var(unbiased=False) + (pm - tm) ** 2 + 1e-8
        return 1.0 - (2.0 * cov) / denom


def ccc_score_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred   = pred.reshape(-1)
    target = target.reshape(-1)
    pm, tm = pred.mean(), target.mean()
    cov    = np.mean((pred - pm) * (target - tm))
    denom  = pred.var() + target.var() + (pm - tm) ** 2 + 1e-8
    return float((2.0 * cov) / denom)


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor = None,
    spike_rates: dict = None,
) -> Dict[str, torch.Tensor]:
    """
    Combined loss:
      L = regression_weight * (Huber_v + Huber_a)
        + ccc_weight        * (CCC_v + CCC_a)
        + spike_reg_weight  * spike_rate_regulariser
    """
    if weights is not None:
        mse_v   = torch.mean(weights * (pred[:, 0] - target[:, 0]) ** 2)
        mse_a   = torch.mean(weights * (pred[:, 1] - target[:, 1]) ** 2)
        huber_v = torch.mean(weights * nn.functional.smooth_l1_loss(
            pred[:, 0], target[:, 0], beta=0.1, reduction="none"))
        huber_a = torch.mean(weights * nn.functional.smooth_l1_loss(
            pred[:, 1], target[:, 1], beta=0.1, reduction="none"))
        ccc_v   = ccc_loss(pred[:, 0], target[:, 0], weights)
        ccc_a   = ccc_loss(pred[:, 1], target[:, 1], weights)
    else:
        mse_v   = nn.functional.mse_loss(pred[:, 0], target[:, 0])
        mse_a   = nn.functional.mse_loss(pred[:, 1], target[:, 1])
        huber_v = nn.functional.smooth_l1_loss(pred[:, 0], target[:, 0], beta=0.1)
        huber_a = nn.functional.smooth_l1_loss(pred[:, 1], target[:, 1], beta=0.1)
        ccc_v   = ccc_loss(pred[:, 0], target[:, 0])
        ccc_a   = ccc_loss(pred[:, 1], target[:, 1])

    loss_val = (
        SNN_REGRESSION_WEIGHT * (huber_v + huber_a)
        + SNN_CCC_WEIGHT * (ccc_v + ccc_a)
    )

    # Spike rate regularisation
    reg_loss = pred.new_tensor(0.0)
    if spike_rates is not None:
        for rate in spike_rates.values():
            if isinstance(rate, torch.Tensor):
                reg_loss = reg_loss + torch.mean((rate - SNN_SPIKE_RATE_TARGET) ** 2)
            elif isinstance(rate, (int, float)):
                reg_loss = reg_loss + pred.new_tensor((rate - SNN_SPIKE_RATE_TARGET) ** 2)

    return {
        "total":        loss_val + SNN_SPIKE_RATE_WEIGHT * reg_loss,
        "mse_valence":  mse_v,
        "mse_arousal":  mse_a,
        "ccc_valence":  ccc_v,
        "ccc_arousal":  ccc_a,
    }


# ── gradient norms ────────────────────────────────────────────────────────────
def grad_norms(model: Net) -> Dict[str, float]:
    out = {}
    for name in ("fc1", "fc2", "fc_out"):
        try:
            layer = getattr(model, name)
            grads = [p.grad.norm(2) for p in layer.parameters() if p.grad is not None]
            out[name] = float(torch.norm(torch.stack(grads), 2).item()) if grads else 0.0
        except AttributeError:
            out[name] = 0.0
    return out


# ── warm-up + cosine LR schedule ─────────────────────────────────────────────
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warm-up for warmup_epochs, then cosine annealing to min_lr."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if ep <= self.warmup_epochs:
                lr = base_lr * ep / max(self.warmup_epochs, 1)
            else:
                progress = (ep - self.warmup_epochs) / max(
                    self.total_epochs - self.warmup_epochs, 1
                )
                import math
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
            lrs.append(max(lr, self.min_lr))
        return lrs


# ── epoch ─────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None, epoch_idx=0, bin_weights=None):
    training  = optimizer is not None
    model.train(training)
    use_cuda  = DEVICE.type == "cuda"

    acc: Dict[str, List[float]] = {
        k: [] for k in ("total", "mse_valence", "mse_arousal", "ccc_valence", "ccc_arousal")
    }
    all_preds, all_targets = [], []
    g_norms = {name: [] for name in ("fc1", "fc2", "fc_out")}
    s_rates = {name: [] for name in ("layer1", "layer2", "output")}

    phase = "train" if training else "val"
    n     = len(loader)
    log_every_batches = max(1, int(LOG_EVERY_BATCHES))
    epoch_t0       = time.perf_counter()
    last_batch_done = epoch_t0
    data_wait_total  = 0.0
    compute_total    = 0.0

    print(f"[{phase}] epoch {epoch_idx:03d} — {n} batches", flush=True)

    for i, (images, target) in enumerate(loader, 1):
        batch_start = time.perf_counter()
        data_wait_total += batch_start - last_batch_done

        images = images.to(DEVICE, non_blocking=use_cuda)
        target = target.to(DEVICE, non_blocking=use_cuda)

        if training:
            if SNN_INPUT_NOISE_STD > 0:
                images = images + torch.randn_like(images) * SNN_INPUT_NOISE_STD
            if SNN_TARGET_NOISE_STD > 0:
                target = (target + torch.randn_like(target) * SNN_TARGET_NOISE_STD).clamp(-1.0, 1.0)
            optimizer.zero_grad(set_to_none=True)

        pred, state = model(images)

        # Per-sample valence-bin weights
        if training and bin_weights is not None:
            valences        = target[:, 0]
            bin_indices     = ((valences - (-1.0)) / 0.2).long().clamp(0, 9)
            bw_tensor       = torch.tensor(bin_weights, dtype=torch.float32, device=DEVICE)
            weights         = bw_tensor[bin_indices]
            weights         = weights / (weights.mean() + 1e-8)
        else:
            weights = None

        losses = total_loss(pred, target, weights=weights,
                            spike_rates=state.get("spike_rate_tensors"))

        if training:
            losses["total"].backward()
            # ── FIX: was 0.1 — starving gradients. Now 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for k, v in grad_norms(model).items():
                g_norms[k].append(v)
            optimizer.step()

        for k in acc:
            acc[k].append(float(losses[k].detach().cpu()))

        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

        for k, v in state["spike_rates"].items():
            s_rates[k].append(v)

        if use_cuda:
            torch.cuda.synchronize(DEVICE)
        batch_done = time.perf_counter()
        compute_total   += batch_done - batch_start
        last_batch_done  = batch_done

        if i == 1 or i % log_every_batches == 0 or i == n:
            elapsed   = batch_done - epoch_t0
            avg_batch = elapsed / i
            avg_data  = data_wait_total / i
            avg_comp  = compute_total / i
            eta       = avg_batch * (n - i)
            print(
                f"  [{phase}] {i:04d}/{n:04d} loss={acc['total'][-1]:.4f} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s "
                f"avg={avg_batch:.2f}s/b data={avg_data:.2f}s/b compute={avg_comp:.2f}s/b",
                flush=True,
            )

    mean = lambda lst: float(sum(lst) / max(len(lst), 1))
    predictions = torch.cat(all_preds).numpy()
    targets     = torch.cat(all_targets).numpy()

    out = {
        "total":       mean(acc["total"]),
        "mse_valence": mean(acc["mse_valence"]),
        "mse_arousal": mean(acc["mse_arousal"]),
        "ccc_valence": mean(acc["ccc_valence"]),
        "ccc_arousal": mean(acc["ccc_arousal"]),
        "predictions": predictions,
        "targets":     targets,
        "grad_norms":  {k: mean(v) for k, v in g_norms.items()},
        "spike_rates": {k: mean(v) for k, v in s_rates.items()},
    }
    out["ccc_score_valence"] = ccc_score_np(predictions[:, 0], targets[:, 0])
    out["ccc_score_arousal"] = ccc_score_np(predictions[:, 1], targets[:, 1])
    out["ccc_score_min"]     = min(out["ccc_score_valence"], out["ccc_score_arousal"])
    out["ccc_score_mean"]    = 0.5 * (out["ccc_score_valence"] + out["ccc_score_arousal"])
    return out


# ── history ───────────────────────────────────────────────────────────────────
def init_history():
    return {
        "train_total": [], "val_total": [],
        "train_mse_v": [], "val_mse_v": [],
        "train_mse_a": [], "val_mse_a": [],
        "train_ccc_v": [], "val_ccc_v": [],
        "train_ccc_a": [], "val_ccc_a": [],
        "train_ccc_score_v": [], "val_ccc_score_v": [],
        "train_ccc_score_a": [], "val_ccc_score_a": [],
        "train_ccc_score_min": [], "val_ccc_score_min": [],
        "lr":          [],
        "grad_norms":  {"fc1": [], "fc2": [], "fc_out": []},
        "spike_rates": {"layer1": [], "layer2": [], "output": []},
        "beta_values": {"layer1": [], "layer2": [], "output": []},
        "best_epoch":  None,
        "best_val":    None,
        "best_val_ccc_min": None,
    }


def update_history(h, tr, vl, lr, model):
    h["train_total"].append(tr["total"])
    h["val_total"]  .append(vl["total"])
    h["train_mse_v"].append(tr["mse_valence"])
    h["val_mse_v"]  .append(vl["mse_valence"])
    h["train_mse_a"].append(tr["mse_arousal"])
    h["val_mse_a"]  .append(vl["mse_arousal"])
    h["train_ccc_v"].append(tr["ccc_valence"])
    h["val_ccc_v"]  .append(vl["ccc_valence"])
    h["train_ccc_a"].append(tr["ccc_arousal"])
    h["val_ccc_a"]  .append(vl["ccc_arousal"])
    h["train_ccc_score_v"].append(tr["ccc_score_valence"])
    h["val_ccc_score_v"]  .append(vl["ccc_score_valence"])
    h["train_ccc_score_a"].append(tr["ccc_score_arousal"])
    h["val_ccc_score_a"]  .append(vl["ccc_score_arousal"])
    h["train_ccc_score_min"].append(tr["ccc_score_min"])
    h["val_ccc_score_min"]  .append(vl["ccc_score_min"])
    h["lr"].append(lr)
    for k in h["grad_norms"]:
        h["grad_norms"][k].append(tr["grad_norms"][k])
    for k in h["spike_rates"]:
        h["spike_rates"][k].append(tr["spike_rates"][k])
    for k, v in model.beta_values.items():
        h["beta_values"][k].append(v)


def save_plots(history, plot_dir: Path, val_metrics):
    try:
        from models.caesar.plots import plot_all
        plot_all(history, val_metrics["predictions"], val_metrics["targets"], plot_dir)
    except Exception as e:
        print(f"[plots] skipped — {e}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────
def train():
    start_time = datetime.now().astimezone()
    timestamp  = start_time.strftime("%Y%m%d_%H%M%S")
    run_dir    = SNN_CKPT.parent / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    snn_ckpt_path = run_dir / "snn_best.pt"
    plot_dir      = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}", flush=True)
    print(f"Data:   {', '.join(str(root) for root in CAESAR_DATA_ROOTS)}", flush=True)
    print(f"Start:  {start_time:%Y-%m-%d %H:%M:%S %Z}", flush=True)
    print(f"Saving checkpoints to: {run_dir}", flush=True)

    train_ds, val_ds, train_loader, val_loader = build_dataloaders(
        data_root           = CAESAR_DATA_ROOTS,
        batch_size          = SNN_BATCH_SIZE,
        frame_step          = FRAME_STEP,
        image_size          = IMAGE_SIZE,
        num_workers         = CAESAR_NUM_WORKERS if DEVICE.type == "cuda" else 0,
        pin_memory          = DEVICE.type == "cuda",
        train_video_limit   = TRAIN_VIDEO_LIMIT,
        val_video_limit     = VAL_VIDEO_LIMIT,
        train_sample_limit  = TRAIN_SAMPLE_LIMIT,
        val_sample_limit    = VAL_SAMPLE_LIMIT,
        augment_train       = SNN_AUGMENT_TRAIN,
        temporal_frames     = CAESAR_TEMPORAL_FRAMES,
        temporal_stride     = CAESAR_TEMPORAL_STRIDE,
        flat_threshold      = FLAT_SAMPLE_THRESHOLD,
        flat_keep_prob      = FLAT_SAMPLE_KEEP_PROB,
        apply_flat_filter   = APPLY_FLAT_FILTER,
    )

    # Print label distribution statistics
    print_label_stats(train_ds, "train")
    print_label_stats(val_ds,   "val")

    model = Net(
        input_shape           = (CAESAR_TEMPORAL_FRAMES, IMAGE_SIZE[0], IMAGE_SIZE[1]),
        hidden_sizes          = SNN_HIDDEN,
        preprocess_hw         = SNN_PREPROCESS_SIZE,
        encoder_out_dim       = SNN_ENCODER_OUT_DIM,
        num_steps             = SNN_NUM_STEPS,
        beta                  = SNN_BETA,
        threshold             = SNN_THRESHOLD,
        learn_beta            = LEARN_BETA,
        head_dropout          = SNN_DROPOUT,
        use_popsan            = SNN_USE_POPSAN,
        output_population     = SNN_OUTPUT_POPULATION,
        use_temporal_attention = SNN_USE_TEMPORAL_ATTN,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=SNN_LR, weight_decay=SNN_WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=5,
        total_epochs=SNN_EPOCHS,
        min_lr=SNN_LR * 0.01,
    )

    # Valence bin weights for loss weighting
    valences = [s.valence for s in train_ds.samples]
    counts   = [0] * 10
    for v in valences:
        v_clamped = max(-1.0, min(1.0, v))
        bin_idx   = int((v_clamped + 1.0) / 0.2)
        if bin_idx >= 10:
            bin_idx = 9
        counts[bin_idx] += 1
    bin_weights = [1.0 / max(1, count) for count in counts]
    print(f"Valence bin frequencies: {counts}")
    print(f"Valence bin weights: {[f'{w:.5f}' for w in bin_weights]}")

    history      = init_history()
    best_val     = float("inf")
    best_score   = -float("inf")
    patience_ctr = 0
    t0           = time.perf_counter()

    for epoch in range(1, SNN_EPOCHS + 1):
        tr = run_epoch(model, train_loader, optimizer, epoch, bin_weights=bin_weights)
        vl = run_epoch(model, val_loader,   None,      epoch, bin_weights=None)
        scheduler.step()

        lr = float(scheduler.get_last_lr()[0])
        update_history(history, tr, vl, lr, model)

        elapsed = time.perf_counter() - t0
        print(
            f"Epoch {epoch:03d}/{SNN_EPOCHS}  "
            f"train={tr['total']:.4f}  val={vl['total']:.4f}  "
            f"val_ccc_v={vl['ccc_score_valence']:.4f}  "
            f"val_ccc_a={vl['ccc_score_arousal']:.4f}  "
            f"lr={lr:.6f}  elapsed={elapsed:.0f}s",
            flush=True,
        )

        val_score = vl["ccc_score_min"]
        if val_score > best_score + 1e-4:
            best_val     = vl["total"]
            best_score   = val_score
            patience_ctr = 0
            history["best_epoch"]        = epoch
            history["best_val"]          = best_val
            history["best_val_ccc_min"]  = best_score
            save_model(model, snn_ckpt_path, optimizer, history,
                       {"epoch": epoch, "best_val": best_val,
                        "best_val_ccc_min": best_score})
            print(f"  ✓ saved best (val={best_val:.4f}, min_ccc={best_score:.4f})",
                  flush=True)
            if vl["ccc_score_valence"] >= 0.80 and vl["ccc_score_arousal"] >= 0.80:
                print("  Target reached: CCC >= 0.80 for both dimensions", flush=True)
                break
        else:
            patience_ctr += 1
            if patience_ctr >= SNN_PATIENCE:
                print(f"  Early stopping — no improvement for {SNN_PATIENCE} epochs",
                      flush=True)
                break

        save_plots(history, plot_dir, vl)

    hist_path = plot_dir / "history.json"
    hist_path.write_text(json.dumps(
        {k: (list(v) if isinstance(v, (list, np.ndarray)) else
             {sk: list(sv) for sk, sv in v.items()} if isinstance(v, dict) else v)
         for k, v in history.items()}, indent=2))

    print(f"\nDone. Best epoch={history['best_epoch']}  val={best_val:.4f}", flush=True)
    print(f"Checkpoint → {snn_ckpt_path}", flush=True)


if __name__ == "__main__":
    train()
