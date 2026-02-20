\
import argparse
import glob
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import ECG_MAE
from datasets import CODE15Dataset


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_target(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert raw signal to patch targets for MSE reconstruction.
    x: (B, C, L)
    returns: (B, N, patch_size*C)
    """
    B, C, L = x.shape
    if L % patch_size != 0:
        raise ValueError("L must be divisible by patch_size.")
    N = L // patch_size
    # (B, C, N, P) -> (B, N, P, C) -> (B, N, P*C)
    target = x.view(B, C, N, patch_size).permute(0, 2, 3, 1).reshape(B, N, patch_size * C)
    return target


def run_pretraining(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seq_len: int = 4096,
    patch_size: int = 64,
    embed_dim: int = 128,
    mask_ratio: float = 0.75,
    num_workers: int = 0,
    seed: int = 42,
    save_dir: str = "./checkpoints",
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    npy_files: List[str] = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    if len(npy_files) == 0:
        raise FileNotFoundError(f"No .npy files found under: {data_dir}")
    print(f"Found {len(npy_files)} .npy files")

    dataset = CODE15Dataset(npy_files, seq_len=seq_len, in_channels=1)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = ECG_MAE(
        seq_len=seq_len,
        in_channels=1,
        patch_size=patch_size,
        embed_dim=embed_dim,
        mask_ratio=mask_ratio,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    os.makedirs(save_dir, exist_ok=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for step, batch_data in enumerate(dataloader, start=1):
            batch_data = batch_data.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            target = build_target(batch_data, patch_size=model.patch_size)

            output = model(batch_data)
            loss = criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            if step % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Step [{step}/{len(dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        ckpt_path = os.path.join(save_dir, f"ecg_mae_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ECG-MAE self-supervised pretraining")
    p.add_argument("--data_dir", type=str, default="./data/code15")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--mask_ratio", type=float, default=0.75)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pretraining(**vars(args))
