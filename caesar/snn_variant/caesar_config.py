"""
Central config — edit here, nowhere else.
All paths, hyperparameters, and flags live in this file.
"""
from pathlib import Path

import torch

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
DATA_ROOT     = ROOT / "datasets/saved_mri_2"
CAESAR_DATA_ROOTS = (
    ROOT / "datasets/saved_mri_2",
)
FRIDAY_ROOT   = Path("/srv/tribe-share/fridaydataset")
SWIFT_CKPT    = ROOT / "hri_affect/checkpoints/swift_best.pt"
SNN_CKPT      = ROOT / "hri_affect/checkpoints/snn_best.pt"
PLOT_DIR      = ROOT / "hri_affect/plots"
LOG_DIR       = ROOT / "hri_affect/logs"
VENV_ROOT     = ROOT / "hri_affect/.venv"

# ── device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── video / data ──────────────────────────────────────────────────────────────
IMAGE_SIZE              = (180, 320)    # (H, W)
SNN_PREPROCESS_SIZE     = (96, 96)      # (H, W) compact CNN projection target
FRAME_STEP              = 6             # 29.97fps / 6 ≈ 5fps
CAESAR_TEMPORAL_FRAMES  = 16            # Temporal stack depth
CAESAR_TEMPORAL_STRIDE  = 2             # Frame spacing inside the stack
TRAIN_SPLIT             = 0.9           # 90% train, 10% val
CAESAR_SPLIT_SEED       = 42
CAESAR_SHUFFLE_SEED     = 1337
# ── CRITICAL PERF FIX: increased from 2 to 4 ──
# Each worker caches its video chunk in RAM after first access.
# 4 workers → 4 chunks preloaded concurrently → much better GPU saturation.
CAESAR_NUM_WORKERS      = 4
LABEL_FILE              = "swiFT_predictions.csv"
VIDEO_FILE              = "source_video.mp4"

# ── Flat-sample filtering ─────────────────────────────────────────────────────
# Samples with |valence| < threshold AND |arousal| < threshold are "flat".
# These are near-neutral and overrepresented. Downsample them.
FLAT_SAMPLE_THRESHOLD   = 0.15          # |v| < 0.15 AND |a| < 0.15 → flat
FLAT_SAMPLE_KEEP_PROB   = 0.5           # Keep 50% of flat samples
APPLY_FLAT_FILTER       = True          # Enable flat-sample filtering

# ── SNN model ─────────────────────────────────────────────────────────────────
SNN_HIDDEN               = (384, 128)
SNN_ENCODER_OUT_DIM      = 512          # Spatial encoder output dimension
SNN_NUM_STEPS            = 4            # BPTT steps
SNN_BETA                 = 0.95
SNN_THRESHOLD            = 0.5
SNN_DROPOUT              = 0.30
LEARN_BETA               = True
SNN_USE_POPSAN           = True         # Population-coded output decoder
SNN_OUTPUT_POPULATION    = 21           # Neurons per output dimension
SNN_USE_TEMPORAL_ATTN    = True         # Temporal step-wise attention pooling

# ── SNN training ──────────────────────────────────────────────────────────────

# ── LR increased from 5e-5 to 3e-4 ──
# With 4 workers the GPU will now be saturated. Use a proper LR.
# Previous 5e-5 was tuned for ~1 effective sample/s. Now we get ~30+.
SNN_LR              = 3e-4
SNN_WEIGHT_DECAY    = 1e-4
SNN_PATIENCE        = 15               # Increased: was triggering too early
SNN_INPUT_NOISE_STD  = 0.02            # Slightly more noise → better regularisation
SNN_TARGET_NOISE_STD = 0.0
SNN_SPIKE_RATE_TARGET = 0.12
SNN_SPIKE_RATE_WEIGHT = 0.01           # Reduced: was competing with CCC loss
SNN_CCC_WEIGHT      = 2.0              # Increased: CCC is the real metric
SNN_REGRESSION_WEIGHT = 0.05





# ── SwiFT model ───────────────────────────────────────────────────────────────
SWIFT_INPUT_DIM   = 20484
SWIFT_HIDDEN_DIM  = 256
SWIFT_NUM_LAYERS  = 4
SWIFT_NUM_HEADS   = 8
SWIFT_DROPOUT     = 0.2
SWIFT_SEQ_LEN     = 20

# ── SwiFT training ────────────────────────────────────────────────────────────
SWIFT_EPOCHS       = 100
SWIFT_BATCH_SIZE   = 8
SWIFT_LR           = 1e-4
SWIFT_WEIGHT_DECAY = 1e-4
SWIFT_PATIENCE     = 80

# ── logging ───────────────────────────────────────────────────────────────────
LOG_EVERY_BATCHES  = 25
DIAG_EVERY_EPOCHS  = 5


class Config:
    """Config object wrapping all module-level constants."""
    pass


_config = Config()
for name, value in list(globals().items()):
    if name.isupper() and not name.startswith("_"):
        setattr(_config, name, value)

config = _config
