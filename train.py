import os
import random
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LightResConv3D_MS_Mamba_Attention
from dataset import compute_extra_metrics, CT3DDataset
from ptflops import get_model_complexity_info
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        probs.append(prob.cpu().numpy())
        preds.append(prob.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)

    results = {
        'ACC': accuracy_score(y_true, y_pred),
        'MacroF1': f1_score(y_true, y_pred, average='macro')
    }
    results.update(compute_extra_metrics(y_true, y_prob))
    return results
def main(args):
    df = pd.read_csv(args.label_csv)  
    df = df.rename(columns={df.columns[0]: 'id', df.columns[1]: 'label'})
    label_counts = df["label"].value_counts()
    can_stratify = (label_counts.min() >= 2)
    if can_stratify:
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df["label"],
            random_state=42
        )
        print("Using stratified split")
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
        print("Stratified split disabled (rare class < 2 samples)")
    train_set = CT3DDataset(img_dir=args.img_dir, label_csv=args.label_csv, df_subset=train_df)
    val_set   = CT3DDataset(img_dir=args.img_dir, label_csv=args.label_csv, df_subset=val_df)
    from torch.utils.data._utils.collate import default_collate
    def collate_skip_none(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return torch.empty(0), torch.empty(0)
        batch = [tuple(b) if isinstance(b, list) else b for b in batch]
        return default_collate(batch)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_skip_none
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_skip_none
    )
    device = torch.device("cuda:0")
    model = LightResConv3D_MS_Mamba_Attention()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device) 
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("\nPer-module parameter count:")
    for name, module in model.named_modules():
        module_params = sum(p.numel() for p in module.parameters(recurse=False))
        if module_params > 0:  
            print(f"{name:40s} : {module_params:,}")
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model, (1, 128, 128, 128), as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
        print(f"FLOPs: {macs}")  # MACs * 2 ≈ FLOPs
        print(f"Params from ptflops: {params}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        metrics = validate(model, val_loader, device)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_csv",
        type=str,
        default="data/label.csv"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/data/pre_dataset"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    main(args=args)
