# Checkpoints

## Included

| File | What it is | Size |
|---|---|---|
| `low_level_locomotion/policy.pt` | Pre-trained **low-level walking policy** for the Go2 (TorchScript export of an `rsl_rl` MLP, 48 inputs -> 12 joint targets). ENMA sends its corrected command to this policy in simulation. | 0.78 MB |
| `low_level_locomotion/policy.onnx` | The same kind of walking policy exported to ONNX. | 0.16 MB |
| `enma_debug_run_2026-07-01/model_0.pt` | An **ENMA policy checkpoint from a short debugging run** (started 2026-07-01 14:00). It has the full ENMA structure (`actor.fc1..fc4`, `actor.lif1..lif4`, `actor.fc_out`, `critic.net.*`) and is useful for checking that the model code loads. **It is not one of the checkpoints evaluated in the paper.** | 1.2 MB |
| `enma_debug_run_2026-07-01/run.log` | Training log of that debugging run. | 1.1 MB |

Load an ENMA checkpoint:

```python
import torch
from ENMAmiddle.model.enma_policy import ENMAPolicy

policy = ENMAPolicy(obs_dim=12, act_dim=6, num_envs=1, device="cpu")
policy.load_state_dict(torch.load("checkpoints/enma_debug_run_2026-07-01/model_0.pt", map_location="cpu"))
```

## Not included (stored on the lab cluster)

| Model | Original location | Where the scripts expect it here |
|---|---|---|
| ENMA (best total return), evaluated in the paper | training run `2026-06-29_10-35-15`, `model_0.pt` | `checkpoints/enma/model_0.pt` |
| ENMA (best affective reward), evaluated in the paper | training run `2026-06-29_10-35-15`, `model_5550.pt` | `checkpoints/enma/model_5550.pt` |
| Caesar student weights | `models/fast_cnn/checkpoints/best_model.pth` | `checkpoints/caesar/best_model.pth` |
| SwiFT-stage and TRIBE v2 weights | see `teacher_pipeline/` | folder given by `$ENMA_TEACHER_ROOT` |
