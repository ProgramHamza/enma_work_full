# Caesar — the student affect model

Caesar is the lightweight video model that gives ENMA its affect reward. It looks at a short clip from the
simulated human's first-person camera and predicts two numbers: **valence** and **arousal**.

## Architecture

```
input  : (B, 16, 180, 320)  grayscale frames, values in [-1, 1]
         -> add channel dim and repeat to 3 channels
         -> torchvision r3d_18 (3D ResNet-18)
         -> final layer replaced by Linear(512 -> 2)
output : (B, 2) = [valence, arousal]
```

The network is defined in two places in this repository:

- [`test_fastcnn.py`](test_fastcnn.py) — builds the model, loads a checkpoint and prints the prediction for a black
  and a mid-grey clip (a quick sanity test).
- [`../ENMAmiddle/training/train_enma.py`](../ENMAmiddle/training/train_enma.py), function `_build_fast_affect_cnn()` — the copy used
  during reinforcement learning. In the code the model is called `FastAffectCNN` / "FastCNN".

Caesar is trained offline on pseudo-labels produced by the teacher pipeline
([`../teacher_pipeline/`](../teacher_pipeline/)). According to the paper the backbone is initialised from Kinetics-400 weights, and on 96
held-out three-minute chunks the student reaches Pearson r = 0.59 (valence) and 0.70 (arousal) against the teacher.
These numbers measure agreement with the teacher's pseudo-labels, not with human ratings.

## What is and is not included

| Item | Status |
|---|---|
| Model definition | Included (see above) |
| Trained weights (`best_model.pth`) | **Not included.** The scripts look for them in `checkpoints/caesar/best_model.pth` (or `--caesar_ckpt` / `$ENMA_CAESAR_CKPT`) |
| Original training script for the 3D ResNet student (`fast_cnn/train.py`) and its dataset class | **Not included.** They stayed on the laboratory cluster and were not part of the original code package either (it was searched in full) |
| Re-implemented training script, [`train_caesar.py`](train_caesar.py) | Included. **It is not the original code**: it was written afterwards from the paper (Section III-F) and from the original files that survived, and it was not used to produce the Caesar weights or any reported number. The comment at the top of the file lists what is taken from those sources and what is an assumption |

Sanity test once the weights are available: `python caesar/test_fastcnn.py checkpoints/caesar/best_model.pth`.

Training and evaluation with the re-implementation (needs `torch`, `torchvision` and `opencv-python`, and a data
folder with one sub-folder per video chunk holding `source_video.mp4` and `swiFT_predictions.csv`):

```
python caesar/train_caesar.py --data_root data/saved_mri --out_dir checkpoints/caesar
python caesar/train_caesar.py --data_root data/saved_mri --eval_only --ckpt checkpoints/caesar/best_model.pth
```

It writes `best_model.pth` in the same format the RL code loads. The data loading, the training loop, the
checkpoint format and the evaluation were run on a small synthetic dataset with a stand-in network; the
`r3d_18` backbone itself was not run on the machine where this file was written, because `torchvision` was not
installed there.

## `snn_variant/`

[`snn_variant/`](snn_variant/) holds the files that were shipped in the original package under "FastCNN_Caesar":

- `caesar_train.py`, `caesar_config.py` — training code and configuration for an **earlier spiking (SNN)
  video-to-valence/arousal model**, with CCC loss, warm-up + cosine learning-rate schedule and EMA validation metrics.
  This is not the 3D ResNet student.
- `caesar_eval.py` — evaluation loop for `FastAffectCNN`. It imports `VideoFrameDataset` and `FastAffectCNN` from
  modules (`.dataset`, `.train`) that are not part of this release, so it does not run as-is.

They are kept for transparency about how the affect predictor evolved.
