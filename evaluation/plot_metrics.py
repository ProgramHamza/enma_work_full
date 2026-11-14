import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import re

def main():
    parser = argparse.ArgumentParser(description="Plot ENMA training metrics.")
    parser.add_argument("--model-idx", type=str, default=None, 
                        help="Index or name of the model/checkpoint directory to plot (e.g. 50, model_50.pt, or YYYY-MM-DD_HH-MM-SS)")
    parser.add_argument("--ckpt-root", type=str, default=os.path.join("ENMAmiddle", "checkpoints"),
                        help="Directory that contains one sub-folder per training run.")
    parser.add_argument("--logs", type=str, nargs="*", default=["train_enma.log"],
                        help="Training log file(s) written by train_enma.py (stdout redirected to a file).")
    args = parser.parse_args()

    # Find the target checkpoint directory
    base_dir = args.ckpt_root
    dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not dirs:
        print(f"No checkpoint directories found in {base_dir}")
        return
        
    if args.model_idx:
        # Match if the user passed a specific directory name or index
        matched_dir = None
        for d in dirs:
            folder_name = os.path.basename(d)
            if args.model_idx in folder_name:
                matched_dir = d
                break
        if matched_dir:
            latest_dir = matched_dir
        else:
            # Check if model index refers to a model checkpoint name or number
            latest_dir = max(dirs, key=os.path.getmtime)
            print(f"Could not find folder matching '{args.model_idx}'. Defaulting to latest run.")
    else:
        latest_dir = max(dirs, key=os.path.getmtime)
        
    print(f"Loading data from run: {latest_dir}")

    log_paths = list(args.logs)
    mean_returns = []
    return_iterations = []
    
    iterations = []
    valences = []
    arousals = []
    step_bonuses = []
    timestamps = [] # Store elapsed minutes

    policy_losses = []
    value_losses = []
    entropies = []
    loss_iterations = []
    
    return_pattern = re.compile(r"return_:\s+min=[-\d\.]+\s+max=[-\d\.]+\s+mean=([-\d\.]+)")
    metrics_pattern = re.compile(r"\[Iter\s+(\d+)/\d+\]\s+Valence:\s+([+\-\d\.]+)\s*\|\s*Arousal:\s+([+\-\d\.]+)\s*\|\s*(?:Mean\s+Step\s+)?Bonus:\s+([+\-\d\.]+)")
    time_pattern = re.compile(r"Training:\s+\d+%\s*\|\s*.*?\s*\[(\d+):(\d+):?(\d+)?<")
    loss_pattern = re.compile(r"policy_loss=([-\d\.]+(?:e[-\d\.]+)?)\s+value_loss=([-\d\.]+(?:e[-\d\.]+)?)\s+entropy=([-\d\.]+(?:e[-\d\.]+)?)")

    target_max_iter = None
    if args.model_idx:
        num_match = re.search(r"\d+", args.model_idx)
        if num_match:
            target_max_iter = int(num_match.group(0))
            print(f"Filtering log parsing up to iteration: {target_max_iter}")

    global_iter_offset = 0
    global_time_offset = 0.0

    for lp in log_paths:
        if not os.path.exists(lp):
            continue
            
        print(f"Parsing log file: {lp} ...")
        current_time_mins = 0.0
        current_iter = 0
        last_iter_in_file = 0
        last_time_in_file = 0.0
        
        with open(lp, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        start_idx = 0
        for i in range(len(lines) - 1, -1, -1):
            if "BEFORE APP LAUNCHER" in lines[i]:
                start_idx = i
                print(f"  Found 'BEFORE APP LAUNCHER' at line {i+1}. Ignoring previous data in this file.")
                break
                
        for line in lines[start_idx:]:
            time_match = time_pattern.search(line)
            if time_match:
                try:
                    h_or_m = int(time_match.group(1))
                    m_or_s = int(time_match.group(2))
                    secs = time_match.group(3)
                    if secs is not None:
                        current_time_mins = h_or_m * 60.0 + m_or_s + int(secs)/60.0
                    else:
                        current_time_mins = h_or_m + m_or_s/60.0
                    last_time_in_file = current_time_mins
                except:
                    pass

            loss_match = loss_pattern.search(line)
            if loss_match:
                try:
                    adjusted_iter = current_iter + global_iter_offset
                    if target_max_iter is None or adjusted_iter <= target_max_iter:
                        policy_losses.append(float(loss_match.group(1)))
                        value_losses.append(float(loss_match.group(2)))
                        entropies.append(float(loss_match.group(3)))
                        loss_iterations.append(adjusted_iter)
                except:
                    pass

            return_match = return_pattern.search(line)
            if return_match:
                try:
                    val = float(return_match.group(1))
                    adjusted_iter = current_iter + global_iter_offset
                    if target_max_iter is None or adjusted_iter <= target_max_iter:
                        mean_returns.append(val)
                        return_iterations.append(adjusted_iter)
                except:
                    pass
                continue
            
            metrics_match = metrics_pattern.search(line)
            if metrics_match:
                try:
                    current_iter = int(metrics_match.group(1))
                    last_iter_in_file = current_iter
                    adjusted_iter = current_iter + global_iter_offset
                    
                    if target_max_iter is None or adjusted_iter <= target_max_iter:
                        iterations.append(adjusted_iter)
                        valences.append(float(metrics_match.group(2)))
                        arousals.append(float(metrics_match.group(3)))
                        step_bonuses.append(float(metrics_match.group(4)))
                        timestamps.append(current_time_mins + global_time_offset)
                except:
                    pass

        # Add offsets for the next log file so they chain together
        if last_iter_in_file > 0:
            global_iter_offset += (last_iter_in_file + 1)
        if last_time_in_file > 0.0:
            global_time_offset += last_time_in_file

    # Plot metrics in a 2x3 grid layout
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    axs = axs.flatten()
    
    # 1. Policy Performance (Mean Return)
    if mean_returns:
        axs[0].plot(return_iterations, mean_returns, label="Mean Return", alpha=0.15, color='blue')
        if len(mean_returns) > 500:
            import numpy as np
            w_500 = 500
            returns_smooth_500 = np.convolve(mean_returns, np.ones(w_500)/w_500, mode='valid')
            axs[0].plot(return_iterations[w_500-1:], returns_smooth_500, label="Mean Return (SMA-500)", color='indigo', linewidth=2.5)
        elif len(mean_returns) > 50:
            import numpy as np
            w_50 = 50
            returns_smooth_50 = np.convolve(mean_returns, np.ones(w_50)/w_50, mode='valid')
            axs[0].plot(return_iterations[w_50-1:], returns_smooth_50, label="Mean Return (SMA-50)", color='indigo', linewidth=2.5)
            
        try:
            import numpy as np
            x_arr = np.array(return_iterations)
            y_arr = np.array(mean_returns)
            slope, _ = np.polyfit(x_arr, y_arr, 1)
            direction = "IMPROVING (Upward)" if slope > 0 else "DECLINING (Downward)"
            poly_fit = np.poly1d(np.polyfit(x_arr, y_arr, 3))(x_arr)
            axs[0].plot(x_arr, poly_fit, color='red', linestyle='--', linewidth=2.5,
                        label=f"Trend (Poly-3) | Slope={slope:.2e} ({direction})")
        except Exception as e:
            print(f"Could not compute return trend line: {e}")

        axs[0].set_title("Policy Performance (Mean Return)", fontsize=12, fontweight='bold')
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Average Return")
        axs[0].legend(loc="upper left")
        axs[0].grid(True, alpha=0.3)
    else:
        axs[0].text(0.5, 0.5, "No return data found yet", ha='center')

    # 2. Valence and Arousal
    if iterations:
        x_axis = timestamps if len(timestamps) == len(iterations) else iterations
        x_label = "Elapsed Minutes" if len(timestamps) == len(iterations) else "Iteration"
        axs[1].plot(x_axis, valences, 'o-', color='green', alpha=0.15, label="Mean Valence")
        axs[1].plot(x_axis, arousals, 'o-', color='purple', alpha=0.15, label="Mean Arousal")
        
        try:
            import numpy as np
            x_arr = np.array(x_axis)
            slope_v, _ = np.polyfit(x_arr, np.array(valences), 1)
            direction_v = "IMPROVING (Upward)" if slope_v > 0 else "DECLINING (Downward)"
            poly_v = np.poly1d(np.polyfit(x_arr, np.array(valences), 3))(x_arr)
            axs[1].plot(x_arr, poly_v, color='darkgreen', linestyle='--', linewidth=2,
                        label=f"Valence Trend (Poly-3) | Slope={slope_v:.2e} ({direction_v})")
                        
            slope_a, _ = np.polyfit(x_arr, np.array(arousals), 1)
            direction_a = "IMPROVING (Upward)" if slope_a > 0 else "DECLINING (Downward)"
            poly_a = np.poly1d(np.polyfit(x_arr, np.array(arousals), 3))(x_arr)
            axs[1].plot(x_arr, poly_a, color='indigo', linestyle='--', linewidth=2,
                        label=f"Arousal Trend (Poly-3) | Slope={slope_a:.2e} ({direction_a})")
        except Exception as e:
            print(f"Could not compute valence/arousal trend lines: {e}")
            
        axs[1].axhline(0, color='red', linestyle='--', alpha=0.5)
        axs[1].set_title("Affect Valence & Arousal", fontsize=12, fontweight='bold')
        axs[1].set_xlabel(x_label)
        axs[1].set_ylabel("Affect Score")
        axs[1].legend(loc="upper left")
        axs[1].grid(True, alpha=0.3)
    else:
        axs[1].text(0.5, 0.5, "No affective data found yet", ha='center')

    # 3. Step Bonuses
    if iterations:
        x_axis = timestamps if len(timestamps) == len(iterations) else iterations
        x_label = "Elapsed Minutes" if len(timestamps) == len(iterations) else "Iteration"
        axs[2].plot(x_axis, step_bonuses, 'o-', color='royalblue', alpha=0.15, label="Mean Step Bonus")
        
        try:
            import numpy as np
            x_arr = np.array(x_axis)
            slope_b, _ = np.polyfit(x_arr, np.array(step_bonuses), 1)
            direction_b = "IMPROVING (Upward)" if slope_b > 0 else "DECLINING (Downward)"
            poly_b = np.poly1d(np.polyfit(x_arr, np.array(step_bonuses), 3))(x_arr)
            axs[2].plot(x_arr, poly_b, color='darkblue', linestyle='--', linewidth=2,
                        label=f"Bonus Trend (Poly-3) | Slope={slope_b:.2e} ({direction_b})")
        except Exception as e:
            print(f"Could not compute bonus trend line: {e}")
            
        axs[2].axhline(0, color='red', linestyle='--', alpha=0.5)
        axs[2].set_title("Affect Step Bonus", fontsize=12, fontweight='bold')
        axs[2].set_xlabel(x_label)
        axs[2].set_ylabel("Reward Value")
        axs[2].legend(loc="upper left")
        axs[2].grid(True, alpha=0.3)
    else:
        axs[2].text(0.5, 0.5, "No step bonus data found yet", ha='center')

    # 4. Action Entropy
    if entropies:
        axs[3].plot(loss_iterations, entropies, label="Entropy", alpha=0.2, color='orange')
        try:
            import numpy as np
            x_arr = np.array(loss_iterations)
            y_arr = np.array(entropies)
            slope_e, _ = np.polyfit(x_arr, y_arr, 1)
            direction_e = "IMPROVING (Upward)" if slope_e > 0 else "DECLINING (Downward)"
            poly_e = np.poly1d(np.polyfit(x_arr, y_arr, 3))(x_arr)
            axs[3].plot(x_arr, poly_e, color='darkorange', linestyle='--', linewidth=2,
                        label=f"Entropy Trend (Poly-3) | Slope={slope_e:.2e} ({direction_e})")
        except Exception as e:
            print(f"Could not compute entropy trend line: {e}")

        axs[3].set_title("Action Entropy (Exploration)", fontsize=12, fontweight='bold')
        axs[3].set_xlabel("Iteration")
        axs[3].set_ylabel("Entropy Value")
        axs[3].legend(loc="upper left")
        axs[3].grid(True, alpha=0.3)
    else:
        axs[3].text(0.5, 0.5, "No entropy data found yet", ha='center')

    # 5. Policy Loss
    if policy_losses:
        axs[4].plot(loss_iterations, policy_losses, label="Policy Loss", alpha=0.2, color='teal')
        try:
            import numpy as np
            x_arr = np.array(loss_iterations)
            y_arr = np.array(policy_losses)
            slope_pl, _ = np.polyfit(x_arr, y_arr, 1)
            direction_pl = "IMPROVING (Upward)" if slope_pl > 0 else "DECLINING (Downward)"
            poly_pl = np.poly1d(np.polyfit(x_arr, y_arr, 3))(x_arr)
            axs[4].plot(x_arr, poly_pl, color='darkcyan', linestyle='--', linewidth=2,
                        label=f"Policy Loss Trend (Poly-3) | Slope={slope_pl:.2e} ({direction_pl})")
        except Exception as e:
            print(f"Could not compute policy loss trend line: {e}")

        axs[4].set_title("Policy Loss", fontsize=12, fontweight='bold')
        axs[4].set_xlabel("Iteration")
        axs[4].set_ylabel("Loss Value")
        axs[4].legend(loc="upper left")
        axs[4].grid(True, alpha=0.3)
    else:
        axs[4].text(0.5, 0.5, "No policy loss data found yet", ha='center')

    # 6. Value Loss
    if value_losses:
        axs[5].plot(loss_iterations, value_losses, label="Value Loss", alpha=0.2, color='crimson')
        try:
            import numpy as np
            x_arr = np.array(loss_iterations)
            y_arr = np.array(value_losses)
            slope_vl, _ = np.polyfit(x_arr, y_arr, 1)
            direction_vl = "IMPROVING (Upward)" if slope_vl > 0 else "DECLINING (Downward)"
            poly_vl = np.poly1d(np.polyfit(x_arr, y_arr, 3))(x_arr)
            axs[5].plot(x_arr, poly_vl, color='darkred', linestyle='--', linewidth=2,
                        label=f"Value Loss Trend (Poly-3) | Slope={slope_vl:.2e} ({direction_vl})")
        except Exception as e:
            print(f"Could not compute value loss trend line: {e}")

        axs[5].set_title("Value Loss", fontsize=12, fontweight='bold')
        axs[5].set_xlabel("Iteration")
        axs[5].set_ylabel("Loss Value")
        axs[5].legend(loc="upper left")
        axs[5].grid(True, alpha=0.3)
    else:
        axs[5].text(0.5, 0.5, "No value loss data found yet", ha='center')

    plt.tight_layout()
    import time
    timestamp = int(time.time())
    output_filename = f"training_metrics_plot_{timestamp}.png"
    output_path = os.path.join(latest_dir, output_filename)
    plt.savefig(output_path, dpi=150)
    print(f"Saved metrics plot layout grid to: {output_path}")
    plt.close()

    # Export to CSV
    try:
        csv_path = os.path.join(latest_dir, f"training_metrics_{timestamp}.csv")
        df_metrics = pd.DataFrame({
            'Iteration': iterations,
            'Valence': valences,
            'Arousal': arousals,
            'Step_Bonus': step_bonuses,
            'Timestamp_Mins': timestamps
        })
        df_metrics.to_csv(csv_path, index=False)
        print(f"Saved metrics CSV data to: {csv_path}")
    except Exception as e:
        print(f"Error exporting CSV: {e}")

if __name__ == "__main__":
    main()