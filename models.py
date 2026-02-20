\
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ECG_MAE(nn.Module):
    """
    1D Masked Autoencoder (MAE) for ECG pretraining.

    Input:  (B, C, L)
    Output: (B, N, patch_size * C) where N = L // patch_size
    """
    def __init__(
        self,
        seq_len: int = 4096,
        in_channels: int = 1,
        patch_size: int = 64,
        embed_dim: int = 128,
        mask_ratio: float = 0.75,
        nhead: int = 4,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError("seq_len must be divisible by patch_size.")
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in (0, 1).")

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

        # Patch embedding: (B, C, L) -> (B, embed_dim, N) -> (B, N, embed_dim)
        self.patch_embed = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Predict raw patch values (flattened)
        self.decoder_pred = nn.Linear(embed_dim, patch_size * in_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def random_masking(
        self, B: int, N: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            ids_keep:    (B, num_keep) indices to keep
            ids_restore: (B, N) indices to restore original order
            mask:        (B, N) 0 for keep, 1 for mask
        """
        num_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]

        mask = torch.ones((B, N), device=device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return ids_keep, ids_restore, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) with L == seq_len
        Returns:
            pred: (B, N, patch_size * C)
        """
        B, C, L = x.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got L={L}.")
        if C != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got C={C}.")

        x_patched = self.patch_embed(x).permute(0, 2, 1)  # (B, N, D)
        x_patched = x_patched + self.pos_embed  # (B, N, D)

        N = x_patched.shape[1]
        ids_keep, ids_restore, _ = self.random_masking(B, N, x.device)

        # Keep a subset of patches
        x_kept = torch.gather(
            x_patched, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        )  # (B, num_keep, D)

        # Append mask tokens, then restore original order
        x_ = torch.cat([x_kept, self.mask_token.repeat(B, N - x_kept.shape[1], 1)], dim=1)
        x_restored = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        )  # (B, N, D)

        encoded = self.transformer(x_restored)  # (B, N, D)
        pred = self.decoder_pred(encoded)       # (B, N, patch_size*C)
        return pred


class ECG_Classifier(nn.Module):
    """
    Transformer classifier for multi-label ECG classification.

    Input:  (B, C, L)
    Output: (B, num_classes) logits
    """
    def __init__(
        self,
        num_classes: int = 5,
        seq_len: int = 4096,
        in_channels: int = 12,
        patch_size: int = 64,
        embed_dim: int = 128,
        nhead: int = 4,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError("seq_len must be divisible by patch_size.")

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got L={L}.")
        if C != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got C={C}.")

        x = self.patch_embed(x).permute(0, 2, 1)  # (B, N, D)
        x = x + self.pos_embed

        encoded = self.transformer(x)   # (B, N, D)
        pooled = encoded.mean(dim=1)    # (B, D)
        logits = self.head(pooled)      # (B, num_classes)
        return logits
