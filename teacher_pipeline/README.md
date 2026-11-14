# Teacher pipeline — TRIBE v2 + SwiFT

The teacher turns ordinary video into valence/arousal **pseudo-labels**. It is slow (tens of seconds per
three-minute chunk), so it is used only offline; the fast student [Caesar](../caesar/) is trained on its output and
replaces it during reinforcement learning.

```
videos -> chunks -> TRIBE v2 (predicted cortical activity) -> SwiFT-style Transformer -> valence / arousal per chunk
```

These scripts adapt existing models and are **not** claimed as novel contributions.

## Files

| File | What it does |
|---|---|
| [`data/youtube_downloader.py`](data/youtube_downloader.py) | Downloads the source videos listed in link files with `yt_dlp`. |
| [`data/video_processor.py`](data/video_processor.py) | Cuts videos into chunks, records them in a manifest and assigns chunks to worker machines. |
| [`tribe_v2/tribe_inference.py`](tribe_v2/tribe_inference.py) | Runs TRIBE v2 (`facebook/tribev2`) over the chunks on several machines (file locks + shared manifest) and stores the predicted brain activity and derived outputs. |
| [`swift/swift_affect_regression.py`](swift/swift_affect_regression.py) | Loads TRIBE v2 and the SwiFT-style model and regresses continuous valence and arousal from the predicted cortical activity; writes predictions and comparison plots. |
| [`data/build_dataset.py`](data/build_dataset.py) | Packages video, brain predictions, ROI time series, features and valence/arousal outputs into one tree: `<output-root>/<video_id>/<chunk_id>/`. |
| [`swift/train_swift.py`](swift/train_swift.py) | **Re-implementation, not the original code.** Trains and evaluates a SwiFT-style Transformer that maps predicted cortical activity to valence and arousal. The original training script and model classes stayed on the laboratory cluster; this file was written afterwards from the paper and the surviving files, and it was not used to produce the teacher weights or any reported number. The comment at the top of the file lists what is taken from those sources and what is an assumption. |

The wrapper that exposes this teacher to the RL code is
[`../ENMAmiddle/rewards/swifttribe_enma_plugin.py`](../ENMAmiddle/rewards/swifttribe_enma_plugin.py).

## Requirements that are not in this repository

- The TRIBE v2 and SwiFT model code and weights (`tribev2`, `models/swift`, `models/swiftold`, `model_swift`).
- A small `config.py` with the cluster paths and the machine list (`CACHE_DIR`, `CODE_DIR`, `MANIFEST`, `MRI_DIR`,
  `CHUNKS_DIR`, `VIDEOS_DIR`, `MACHINES`), imported by `tribe_inference.py` and `video_processor.py`.
- The source videos and the generated dataset.

## Reported teacher accuracy

In the paper the SwiFT stage is evaluated on 121 held-out clips of the *Friday Night Lights* dataset
(Chang et al., 2021): Pearson r = 0.774 (valence) and 0.861 (arousal).
