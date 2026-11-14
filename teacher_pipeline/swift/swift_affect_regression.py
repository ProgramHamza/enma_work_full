import argparse
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import subprocess
import json
from pathlib import Path
from tempfile import TemporaryDirectory

# Add paths to enable imports
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir / "models" / "tribev2"))
sys.path.append(str(root_dir / "models" / "swift"))
sys.path.append(str(root_dir / "models" / "swiftold"))

from tribev2.demo_utils import TribeModel
from model_swift import create_swift_model
from models.swiftold.model import fMRIEmotionTransformer
from models.swift.model import SwiFTModel

def get_video_info(video_path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,nb_frames", "-of", "json", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info['streams'][0]
    num_frames = int(stream['nb_frames'])
    num, den = map(int, stream['r_frame_rate'].split('/'))
    fps = num / den
    return fps, num_frames

def _benchmark_df(timing_results):
    if not timing_results:
        return pd.DataFrame(
            columns=[
                "condition",
                "window_index",
                "time_sec",
                "valence",
                "arousal",
                "get_events_sec",
                "filter_sec",
                "predict_sec",
                "model_forward_sec",
                "total_sec",
            ]
        )
    return pd.DataFrame(timing_results)

def write_benchmark_csv(root_dir, timing_results):
    timing_df = _benchmark_df(timing_results)
    timing_df.to_csv(root_dir / "benchmark_results.csv", index=False)
    return timing_df

def write_live_plot(root_dir, preds_by_window):
    live_plot_path = root_dir / "benchmark_live.png"
    if not preds_by_window:
        live_plot_path.touch(exist_ok=True)
        return
    live_df = pd.concat(preds_by_window, ignore_index=True)
    plot_all(live_df, None, live_plot_path)

def write_benchmark_comparison(root_dir, timing_results):
    timing_df = _benchmark_df(timing_results)
    if timing_df.empty:
        (root_dir / "benchmark_report.md").write_text("No benchmark rows were recorded.\n", encoding="utf-8")
        return

    conditions = list(timing_df["condition"].dropna().unique())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    condition_counts = timing_df.groupby("condition").size()

    flat_axes = list(axes.flat)
    for ax in flat_axes:
        ax.set_visible(False)

    for idx, condition in enumerate(conditions[:4]):
        ax = flat_axes[idx]
        ax.set_visible(True)
        subset = timing_df[timing_df["condition"] == condition].sort_values("time_sec")
        ax.plot(subset["valence"], subset["time_sec"], marker="o", label="valence")
        ax.plot(subset["arousal"], subset["time_sec"], marker="o", linestyle="--", label="arousal")
        ax.set_title(condition)
        ax.set_xlabel("Score")
        ax.set_ylabel("Time (s)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(root_dir / "benchmark_comparison.png")
    plt.close(fig)

    report_lines = [
        "# Benchmark Report",
        "",
    ]
    for condition in conditions:
        report_lines.extend(
            [
                f"## {condition}",
                f"- windows: {int(condition_counts[condition])}",
                f"- total wall-clock: {timing_df[timing_df['condition'] == condition]['total_sec'].sum():.3f} s",
                f"- avg time/window: {timing_df[timing_df['condition'] == condition]['total_sec'].mean():.3f} s",
                "- dominant stage: n/a",
            ]
        )
        if condition == "no_text_no_transcription":
            report_lines.append("- skips transcription: yes")
        elif condition == "no_text":
            report_lines.append("- skips transcription: no")
        else:
            report_lines.append("- skips transcription: unknown")
        report_lines.append("")
    report_lines.extend(
        [
            "Fastest condition: unavailable until all conditions are present.",
            "",
            "Conditions that could not run: none recorded in this benchmark file.",
            "",
        ]
    )
    (root_dir / "benchmark_report.md").write_text("\n".join(report_lines), encoding="utf-8")

def run_windowed_inference(
    video_path,
    checkpoint_path,
    window_duration=2.0,
    device="cpu",
    condition="baseline",
    skip_transcription=False,
    max_windows=None,
):
    fps, total_frames = get_video_info(video_path)
    window_size_frames = int(window_duration * fps)
    
    # 1. Load Tribe
    tribe = TribeModel.from_pretrained("facebook/tribev2", device="cpu")
    
    # 2. Model Detection & Inference
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    
    if "backbone.surface_proj.weight" in state_dict:
        print("Detected SwiFT Surface architecture.")
        embed_dim = int(state_dict["backbone.surface_proj.weight"].shape[0])
        model = create_swift_model(in_channels=1, embed_dim=embed_dim, use_surface_input=True).to(device)
    elif "input_projection.weight" in state_dict:
        print("Detected ROI transformer architecture.")
        input_dim = int(state_dict["input_projection.weight"].shape[1])
        model = fMRIEmotionTransformer(input_dim=input_dim).to(device)
    elif "feature_extractor.weight" in state_dict:
        print("Detected simple SwiFT architecture.")
        input_dim = int(state_dict["feature_extractor.weight"].shape[1])
        model = SwiFTModel(input_dim=input_dim).to(device)
    else:
        raise RuntimeError("Unknown architecture")
        
    model.load_state_dict(state_dict)
    model.eval()
    
    all_preds = []
    timing_results = []
    per_window_preds = []
    
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for i, start_frame in enumerate(range(0, total_frames, window_size_frames)):
            if max_windows is not None and i >= max_windows:
                print(f"Stopping after {max_windows} windows by request.", flush=True)
                break
            window_video = tmp_path / f"window_{i}.mp4"
            os.system(f"ffmpeg -y -ss {start_frame/fps} -t {window_duration} -i {video_path} -c copy {window_video} > /dev/null 2>&1")
            
            # --- Timing Instrumentation ---
            t0 = time.perf_counter()
            df = tribe.get_events_dataframe(video_path=str(window_video), audio_only=skip_transcription)
            t1 = time.perf_counter()
            
            if "type" in df.columns:
                df_filtered = df[~df.type.isin(["Word", "Text", "Sentence"])]
            else:
                df_filtered = df
            t2 = time.perf_counter()

            brain_preds, _ = tribe.predict(events=df_filtered, verbose=False)
            t3 = time.perf_counter()
            
            x = torch.from_numpy(brain_preds.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                y = model(x)
            t4 = time.perf_counter()
            
            pred = y.cpu().numpy().squeeze()
            pred = np.atleast_2d(pred)
            mean_valence = float(pred[:, 0].mean())
            mean_arousal = float(pred[:, 1].mean())
            time_sec = (start_frame / fps) + (window_duration / 2.0)
            all_preds.append(pred)
            per_window_preds.append(
                pd.DataFrame(
                    {
                        "condition": [condition],
                        "window_index": [i],
                        "time_sec": [time_sec],
                        "prediction_index": [0],
                        "valence": [mean_valence],
                        "arousal": [mean_arousal],
                    }
                )
            )
            
            window_timings = {
                "condition": condition,
                "window_index": i,
                "time_sec": time_sec,
                "valence": mean_valence,
                "arousal": mean_arousal,
                "get_events_sec": t1 - t0,
                "filter_sec": t2 - t1,
                "predict_sec": t3 - t2,
                "model_forward_sec": t4 - t3,
                "total_sec": t4 - t0
            }
            timing_results.append(window_timings)
            write_benchmark_csv(root_dir, timing_results)
            write_live_plot(root_dir, per_window_preds)
            write_benchmark_comparison(root_dir, timing_results)
            print(
                "TIMING "
                f"condition={condition} "
                f"window_index={i} "
                f"get_events_sec={window_timings['get_events_sec']:.6f} "
                f"filter_sec={window_timings['filter_sec']:.6f} "
                f"predict_sec={window_timings['predict_sec']:.6f} "
                f"model_forward_sec={window_timings['model_forward_sec']:.6f} "
                f"total_sec={window_timings['total_sec']:.6f}",
                flush=True,
            )
            
    preds = np.concatenate(all_preds, axis=0)
    
    timing_df = write_benchmark_csv(root_dir, timing_results)
    write_live_plot(root_dir, per_window_preds)
    write_benchmark_comparison(root_dir, timing_results)
        
    return pd.DataFrame({'valence': preds[:, 0], 'arousal': preds[:, 1]})

def plot_all(pred_df, ground_truth_path, output_path):
    if "time_sec" in pred_df.columns:
        y = pred_df["time_sec"]
    else:
        y = np.arange(len(pred_df))

    plt.figure(figsize=(8, 8))
    plt.plot(pred_df['valence'], y, label='Pred Valence')
    plt.plot(pred_df['arousal'], y, label='Pred Arousal')
    
    if ground_truth_path and Path(ground_truth_path).exists():
        gt_df = pd.read_csv(ground_truth_path)
        v_col = 'valence' if 'valence' in gt_df.columns else 'valence_pred'
        a_col = 'arousal' if 'arousal' in gt_df.columns else 'arousal_pred'
        gt_y = gt_df["time_sec"] if "time_sec" in gt_df.columns else np.arange(len(gt_df))
        plt.plot(gt_df[v_col], gt_y, label='GT Valence', linestyle='--')
        plt.plot(gt_df[a_col], gt_y, label='GT Arousal', linestyle='--')
        
    plt.legend()
    plt.xlabel("Score")
    plt.ylabel("Time (s)")
    plt.title("Prediction vs Ground Truth")
    plt.savefig(output_path)
    plt.close()
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", default=str(root_dir / "models" / "swiftold" / "checkpoints" / "best_model.pth"))
    parser.add_argument("--gt", help="Ground truth CSV")
    parser.add_argument("--out", default="./comparison.png")
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--condition", type=str, default="baseline", help="A label for the benchmark run.")
    parser.add_argument("--skip-transcription", action="store_true", help="If set, skip the transcription step.")
    parser.add_argument("--max-windows", type=int, default=None, help="Optional cap for validation runs.")
    args = parser.parse_args()
    
    preds = run_windowed_inference(
        args.video, 
        args.checkpoint, 
        window_duration=args.window, 
        condition=args.condition, 
        skip_transcription=args.skip_transcription,
        max_windows=args.max_windows
    )
    plot_all(preds, args.gt, args.out)
