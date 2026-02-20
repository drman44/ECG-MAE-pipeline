\
import ast
import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb


def _to_channels_first(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array shapes to (C, L).
    Accepts (L,), (L,C), (C,L).
    """
    if arr.ndim == 1:
        arr = arr[None, :]  # (1, L)
    elif arr.ndim == 2:
        # Heuristic: if first dim looks like time axis (very long), transpose to (C,L)
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    return arr


def _fix_length(x: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Ensure last dimension is exactly seq_len via truncation or zero-padding.
    x: (C, L)
    """
    C, L = x.shape
    if L >= seq_len:
        return x[:, :seq_len]
    pad = np.zeros((C, seq_len - L), dtype=x.dtype)
    return np.concatenate([x, pad], axis=1)


class CODE15Dataset(Dataset):
    """
    Loads .npy ECG arrays for self-supervised pretraining.
    Expects samples convertible to shape (C, L).

    If a sample has more than `in_channels` channels, it keeps the first `in_channels`.
    If it has fewer, it zero-pads channels.
    """
    def __init__(self, file_paths: Sequence[str], seq_len: int = 4096, in_channels: int = 1):
        self.file_paths = list(file_paths)
        self.seq_len = int(seq_len)
        self.in_channels = int(in_channels)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = np.load(self.file_paths[idx])
        data = _to_channels_first(data)
        data = _fix_length(data, self.seq_len)

        # Fix channel count to in_channels
        C, L = data.shape
        if C > self.in_channels:
            data = data[: self.in_channels, :]
        elif C < self.in_channels:
            pad = np.zeros((self.in_channels - C, L), dtype=data.dtype)
            data = np.concatenate([data, pad], axis=0)

        return torch.tensor(data, dtype=torch.float32)


class PTBXLDataset(Dataset):
    """
    PTB-XL dataset reader via wfdb.
    Returns:
        x: (12, 4096) float32
        y: (num_classes,) float32
    """
    def __init__(self, df: pd.DataFrame, base_dir: str, labels: np.ndarray, seq_len: int = 4096):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.labels = labels
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        relative_path = self.df.loc[idx, "filename_hr"]
        full_path = os.path.join(self.base_dir, relative_path)

        record_data, _ = wfdb.rdsamp(full_path)  # (L, 12)
        if record_data.ndim != 2 or record_data.shape[1] != 12:
            raise ValueError(f"Unexpected PTB-XL shape: {record_data.shape} for {full_path}")

        # (L, 12) -> (12, L)
        record_data = record_data.T
        record_data = _fix_length(record_data, self.seq_len)

        x = torch.tensor(record_data, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def get_superclasses(scp_dict_str: str, diag_df: pd.DataFrame) -> List[str]:
    """
    Map PTB-XL scp_codes string to diagnostic superclasses.
    """
    scp_dict = ast.literal_eval(scp_dict_str)
    superclasses = set()
    for key in scp_dict.keys():
        if key in diag_df.index:
            sc = diag_df.loc[key, "diagnostic_class"]
            if pd.notna(sc):
                superclasses.add(str(sc))
    return list(superclasses)
