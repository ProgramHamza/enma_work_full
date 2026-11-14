import os
import sys
import torch
from pathlib import Path

# Usage: python caesar/test_fastcnn.py [path/to/best_model.pth]
# Default: $ENMA_CAESAR_CKPT, then <repo>/checkpoints/caesar/best_model.pth
_FAST_CNN_CKPT = Path(
    (sys.argv[1] if len(sys.argv) > 1 else None)
    or os.environ.get("ENMA_CAESAR_CKPT")
    or str(Path(__file__).resolve().parents[1] / "checkpoints" / "caesar" / "best_model.pth")
)

def _build_fast_affect_cnn():
    from torchvision.models.video import r3d_18
    import torch.nn as nn
    class FastAffectCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = r3d_18(weights=None)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
        def forward(self, x):
            x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            return self.backbone(x)
    return FastAffectCNN()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = _build_fast_affect_cnn().to(device)
raw = torch.load(_FAST_CNN_CKPT, map_location=device)
state = raw.get("model_state_dict", raw)
state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
state = {f"backbone.{k}": v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model.eval()

with torch.no_grad():
    x = torch.zeros(1, 16, 180, 320, device=device) - 1.0 # black screen is -1.0
    pred = model(x)
    print("Pred on black screen (-1.0):", pred)

    x2 = torch.zeros(1, 16, 180, 320, device=device)
    pred2 = model(x2)
    print("Pred on zero screen (0.0):", pred2)
