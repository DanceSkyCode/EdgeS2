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
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
class CT3DDataset(Dataset):
    def __init__(self, img_dir, label_csv, df_subset=None):
        self.img_dir = img_dir
        self.labels_df = pd.read_csv(label_csv)
        self.labels_df = self.labels_df.rename(columns={self.labels_df.columns[0]: 'id',
                                                        self.labels_df.columns[1]: 'label'})
        self.labels_dict = dict(zip(self.labels_df['id'], self.labels_df['label']))
        if df_subset is not None:
            allowed_prefixes = set(df_subset['id'].tolist())
        else:
            allowed_prefixes = None
        self.samples = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if not f.endswith('.nii.gz'):
                    continue
                prefix = f.replace('.nii.gz', '')
                if prefix not in self.labels_dict:
                    print(f"Label not found, skipped: {f}")
                    continue
                if allowed_prefixes is not None and prefix not in allowed_prefixes:
                    continue 
                self.samples.append((os.path.join(root, f), self.labels_dict[prefix]))

        print(f"Total valid samples found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            img = nib.load(img_path).get_fdata()
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        except Exception as e:
            print(f"Failed to load {img_path}, skipping: {e}")
            return None  #

        return img, label


def compute_extra_metrics(y_true, y_prob, num_classes=4):
    metrics = {}

    # AUROC
    metrics['AUROC'] = roc_auc_score(
        y_true, y_prob, multi_class='ovr', average='macro'
    )

    recalls, specs, accs = [], [], []

    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        prob = y_prob[:, c]

        thresholds = np.linspace(0, 1, 200)
        for t in thresholds:
            y_pred = (prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_bin, y_pred).ravel()

            recall = tp / (tp + fn + 1e-6)
            spec = tn / (tn + fp + 1e-6)
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)

            recalls.append(recall)
            specs.append(spec)
            accs.append(acc)

    recalls = np.array(recalls)
    specs = np.array(specs)
    accs = np.array(accs)

    def max_acc(mask):
        return accs[mask].max() if mask.any() else 0.0

    metrics['A@R75'] = max_acc(recalls >= 0.75)
    metrics['A@S75'] = max_acc(specs >= 0.75)
    metrics['S@R75'] = specs[recalls >= 0.75].max()
    metrics['R@S75'] = recalls[specs >= 0.75].max()

    return metrics
