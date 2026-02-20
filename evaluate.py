\
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
)

from models import ECG_Classifier
from datasets import PTBXLDataset, get_superclasses


PTBXL_SUPERCLASSES: List[str] = ["CD", "HYP", "MI", "NORM", "STTC"]


def safe_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # Ensure 2x2 output even if one class is missing
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def evaluate_model(
    ptbxl_dir: str,
    model_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    out_csv: str = "table2_test_metrics.csv",
    out_png: str = "roc_curve.png",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    db_file = f"{ptbxl_dir}/ptbxl_database.csv"
    scp_df = pd.read_csv(f"{ptbxl_dir}/scp_statements.csv", index_col=0)
    diag_df = scp_df[scp_df["diagnostic"] == 1]

    df_ptbxl = pd.read_csv(db_file, index_col="ecg_id")
    df_test = df_ptbxl[df_ptbxl["strat_fold"] == 10].copy()
    df_test["superclass"] = df_test["scp_codes"].apply(lambda x: get_superclasses(x, diag_df))

    mlb = MultiLabelBinarizer(classes=PTBXL_SUPERCLASSES)
    y_test = mlb.fit_transform(df_test["superclass"])

    test_dl = DataLoader(
        PTBXLDataset(df_test, ptbxl_dir, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = ECG_Classifier(num_classes=len(PTBXL_SUPERCLASSES), in_channels=12).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    probs_list, targets_list = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
            targets_list.append(y.numpy())

    test_probs = np.vstack(probs_list)
    test_targets = np.vstack(targets_list)

    # 1) Metrics table
    metrics: List[Dict[str, float]] = []
    roc_data = []  # for plotting

    for i, c in enumerate(PTBXL_SUPERCLASSES):
        y_true = test_targets[:, i]
        y_prob = test_probs[:, i]

        if np.unique(y_true).size < 2:
            # AUROC/ROC undefined
            metrics.append(
                {
                    "Class": c,
                    "AUROC": np.nan,
                    "AUPRC": float(average_precision_score(y_true, y_prob)),
                    "Sensitivity": np.nan,
                    "Specificity": np.nan,
                    "F1-Score": np.nan,
                }
            )
            continue

        fpr, tpr, thresh = roc_curve(y_true, y_prob)
        auroc = float(auc(fpr, tpr))
        auprc = float(average_precision_score(y_true, y_prob))

        # Youden's J for operating threshold
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        opt_th = float(thresh[best_idx])

        y_pred = (y_prob >= opt_th).astype(int)
        cm = safe_confusion(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        f1 = float(f1_score(y_true, y_pred)) if np.unique(y_pred).size > 1 else 0.0

        metrics.append(
            {
                "Class": c,
                "AUROC": round(auroc, 4),
                "AUPRC": round(auprc, 4),
                "Sensitivity": round(float(sensitivity), 4) if np.isfinite(sensitivity) else np.nan,
                "Specificity": round(float(specificity), 4) if np.isfinite(specificity) else np.nan,
                "F1-Score": round(f1, 4),
                "OptThreshold(YoudenJ)": round(opt_th, 4),
            }
        )
        roc_data.append((c, fpr, tpr, auroc))

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(out_csv, index=False)
    print(f"Metrics saved to {out_csv}")

    # 2) ROC plot
    plt.figure(figsize=(9, 7))
    for c, fpr, tpr, auroc in roc_data:
        plt.plot(fpr, tpr, lw=2.0, label=f"{c} (AUC = {auroc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("PTB-XL Test Set (Fold 10) ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"ROC curve saved to {out_png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned model on PTB-XL fold 10")
    p.add_argument("--ptbxl_dir", type=str, default="./data/ptbxl")
    p.add_argument("--model_path", type=str, default="best_ft_model.pth")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_csv", type=str, default="table2_test_metrics.csv")
    p.add_argument("--out_png", type=str, default="roc_curve.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(**vars(args))
