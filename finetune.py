\
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from models import ECG_Classifier
from datasets import PTBXLDataset, get_superclasses


PTBXL_SUPERCLASSES: List[str] = ["CD", "HYP", "MI", "NORM", "STTC"]


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Robust macro-AUROC: compute per-class AUROC when defined, then average.
    Returns NaN if none of the classes are computable.
    """
    aucs = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        # AUROC undefined if only one class present
        if np.unique(yt).size < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, yp))
        except Exception:
            continue
    if len(aucs) == 0:
        return float("nan")
    return float(np.mean(aucs))


def load_ptbxl_splits(ptbxl_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db_file = f"{ptbxl_dir}/ptbxl_database.csv"
    scp_df = pd.read_csv(f"{ptbxl_dir}/scp_statements.csv", index_col=0)
    diag_df = scp_df[scp_df["diagnostic"] == 1]

    df_ptbxl = pd.read_csv(db_file, index_col="ecg_id")
    df_ptbxl = df_ptbxl.copy()
    df_ptbxl["superclass"] = df_ptbxl["scp_codes"].apply(lambda x: get_superclasses(x, diag_df))

    df_train = df_ptbxl[df_ptbxl["strat_fold"] <= 8].copy()
    df_val = df_ptbxl[df_ptbxl["strat_fold"] == 9].copy()
    df_test = df_ptbxl[df_ptbxl["strat_fold"] == 10].copy()
    return df_train, df_val, df_test


def run_finetuning(
    ptbxl_dir: str,
    pretrained_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    num_workers: int = 0,
    seed: int = 42,
    out_path: str = "best_ft_model.pth",
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_train, df_val, _ = load_ptbxl_splits(ptbxl_dir)

    mlb = MultiLabelBinarizer(classes=PTBXL_SUPERCLASSES)
    y_train = mlb.fit_transform(df_train["superclass"])
    y_val = mlb.transform(df_val["superclass"])

    train_dl = DataLoader(
        PTBXLDataset(df_train, ptbxl_dir, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_dl = DataLoader(
        PTBXLDataset(df_val, ptbxl_dir, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = ECG_Classifier(num_classes=len(PTBXL_SUPERCLASSES), in_channels=12).to(device)

    # Load pretrained weights (partial, shape-safe)
    pretrained_weights = torch.load(pretrained_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in pretrained_weights.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} tensors from pretrained checkpoint")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += float(loss.item())

        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(y.numpy())

        val_probs = np.vstack(val_probs)
        val_targets = np.vstack(val_targets)

        val_auc = macro_auroc(val_targets, val_probs)
        avg_train_loss = train_loss / max(1, len(train_dl))
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Macro AUROC: {val_auc:.4f}")

        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), out_path)
            print(f"Saved best model to: {out_path} (Val Macro AUROC={best_auc:.4f})")

    if best_auc < 0:
        print("Warning: AUROC could not be computed on validation set (class imbalance / no positives).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ECG classifier on PTB-XL")
    p.add_argument("--ptbxl_dir", type=str, default="./data/ptbxl")
    p.add_argument("--pretrained_path", type=str, default="./checkpoints/ecg_mae_epoch_10.pth")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_path", type=str, default="best_ft_model.pth")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_finetuning(**vars(args))
