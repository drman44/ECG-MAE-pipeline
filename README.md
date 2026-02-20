\
# ECG-MAE: Label-Efficient Foundation Model for ECG Classification

Official PyTorch implementation of a **1D Masked Autoencoder (1D-MAE)** foundation model for electrocardiogram (ECG) analysis. This repository provides an end-to-end pipeline for:

- Self-supervised **pretraining** (MAE reconstruction) on unlabelled ECG signals (e.g., CODE-15% exported as `.npy`)
- Supervised **fine-tuning** on **PTB-XL** for multi-label diagnostic superclass classification
- **Evaluation** on PTB-XL official test fold (Fold 10), exporting a metrics table and ROC curves

> Note: This codebase assumes a fixed input length of **4096 samples** and **patch size 64** by default.

---

## 1) Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2) Data layout

Create this structure:

```
data/
  code15/        # pretraining .npy files (any nested folders ok)
  ptbxl/         # PTB-XL root (ptbxl_database.csv, scp_statements.csv, and WFDB files)
```

### Pretraining data (`./data/code15/`)
- Place **.npy** files under `./data/code15/` (subfolders allowed).
- Each `.npy` should contain an ECG array convertible to shape **(C, L)** or **(L, C)** or **(L,)**.
- The loader will **pad/truncate to 4096** and enforce **1 channel** by default.

### Fine-tuning & evaluation data (`./data/ptbxl/`)
- Download PTB-XL and extract it into `./data/ptbxl/` so that the following exist:

```
data/ptbxl/ptbxl_database.csv
data/ptbxl/scp_statements.csv
data/ptbxl/records100/...
data/ptbxl/records500/...
```

---

## 3) Usage

### A) Self-supervised pretraining
```bash
python pretrain.py --data_dir ./data/code15 --epochs 10 --batch_size 64 --save_dir ./checkpoints
```

This writes checkpoints like:
- `./checkpoints/ecg_mae_epoch_1.pth`
- ...
- `./checkpoints/ecg_mae_epoch_10.pth`

### B) Supervised fine-tuning (PTB-XL)
```bash
python finetune.py --ptbxl_dir ./data/ptbxl --pretrained_path ./checkpoints/ecg_mae_epoch_10.pth --epochs 10
```

Outputs:
- `best_ft_model.pth` (best validation macro-AUROC)

### C) Evaluation (PTB-XL official test fold)
```bash
python evaluate.py --ptbxl_dir ./data/ptbxl --model_path best_ft_model.pth
```

Outputs:
- `table2_test_metrics.csv`
- `roc_curve.png`

---

## 4) Reproducibility notes
- Random seeds are set in `pretrain.py` and `finetune.py` (default: 42).
- For Windows stability, default DataLoader `num_workers=0`. You can increase it if your setup supports it.

---

## 5) Citation
If you use this code in your research, please cite your accompanying manuscript:

```
(Citation details will be added upon publication)
```
