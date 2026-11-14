import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Import elements directly from local package layers
from .dataset import VideoFrameDataset
from .train import FastAffectCNN, ccc_loss

def evaluate():
    base_dir = Path("/mnt/tribe-share/enma")
    
    # Target directory for evaluation
    val_dir = base_dir / "datasets/saved_mri_2"
    
    print("Loading evaluation dataset...", flush=True)
    # Force split="train" so it reads 100% of the files within this specific target directory
    val_dataset = VideoFrameDataset(
        [val_dir], 
        split="train", 
        augment=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        num_workers=4,
        pin_memory=True
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FastAffectCNN().to(device)
    
    ckpt_path = base_dir / "models/fast_cnn/checkpoints/best_model.pth"
    print(f"Loading checkpoint weights from: {ckpt_path}", flush=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc="Evaluating", unit="batch", leave=True)
    
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            
            # --- FIXED: Match training normalization perfectly ---
            # Cast to float and compress 0-255 range down to [-1.0, 1.0]
            imgs = imgs.float() / 255.0 * 2.0 - 1.0
            
            # Run mixed precision inference inside standard AMP framework
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                preds = model(imgs)
                
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate Mean Absolute Error
    mae_v = torch.mean(torch.abs(all_preds[:, 0] - all_labels[:, 0])).item()
    mae_a = torch.mean(torch.abs(all_preds[:, 1] - all_labels[:, 1])).item()
    
    # Calculate Concordance Correlation Coefficient (CCC)
    ccc_v = 1.0 - ccc_loss(all_preds[:, 0], all_labels[:, 0]).item()
    ccc_a = 1.0 - ccc_loss(all_preds[:, 1], all_labels[:, 1]).item()
    
    # Pearson Product-Moment Correlation Function
    def pearson(x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        
    pearson_v = pearson(all_preds[:, 0], all_labels[:, 0]).item()
    pearson_a = pearson(all_preds[:, 1], all_labels[:, 1]).item()
    
    print("\n" + "="*50)
    print(f"=== FINAL EVALUATION METRICS ({val_dir.name}) ===")
    print("="*50)
    print(f"Valence CCC:     {ccc_v:.4f} | Arousal CCC:     {ccc_a:.4f}")
    print(f"Valence Pearson: {pearson_v:.4f} | Arousal Pearson: {pearson_a:.4f}")
    print(f"Valence MAE:     {mae_v:.4f} | Arousal MAE:     {mae_a:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate()