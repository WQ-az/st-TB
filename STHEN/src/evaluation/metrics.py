import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from typing import Dict
from tqdm import tqdm


@torch.no_grad()
def evaluate_temporal_localization(model: nn.Module, test_loader: DataLoader,
                                   device: torch.device) -> Dict:
    model.eval()
    all_pred_tcp = []
    all_true_tcp = []
    peak_errors = []

    for batch in tqdm(test_loader, desc="Evaluating Temporal Localization"):
        batch = batch.to(device)
        outputs = model(batch)
        tcp_score = outputs['tcp_score']
        tcp_label = batch.tcp_label
        batch_idx = batch.batch
        labels = batch.y

        B, T = tcp_label.shape
        pred_tcp = torch.zeros(B, T, device=device)
        for g in range(B):
            mask = (batch_idx == g)
            if mask.sum() > 0:
                pred_tcp[g] = tcp_score[mask].mean(dim=0)

        all_pred_tcp.append(pred_tcp.cpu().numpy())
        all_true_tcp.append(tcp_label.cpu().numpy())

        for g in range(B):
            if labels[g].item() == 0:
                continue
            true_peak = tcp_label[g].argmax().item()
            pred_peak = pred_tcp[g].argmax().item()
            peak_errors.append(abs(pred_peak - true_peak))

    all_pred_tcp = np.concatenate(all_pred_tcp, axis=0)
    all_true_tcp = np.concatenate(all_true_tcp, axis=0)

    metrics = {
        'peak_time_error_mean': np.mean(peak_errors) if peak_errors else 0.0,
        'peak_time_error_std': np.std(peak_errors) if peak_errors else 0.0,
    }

    try:
        y_true_bin = (all_true_tcp > 0.5).astype(int).flatten()
        y_pred_bin = all_pred_tcp.flatten()
        metrics['tcp_auc'] = roc_auc_score(y_true_bin, y_pred_bin)
    except Exception:
        metrics['tcp_auc'] = np.nan

    return metrics


@torch.no_grad()
def evaluate_spatial_localization(model: nn.Module, test_loader: DataLoader,
                                  device: torch.device) -> Dict:
    model.eval()
    center_offsets = []

    for batch in tqdm(test_loader, desc="Evaluating Spatial Localization"):
        batch = batch.to(device)
        outputs = model(batch)
        scm_score = outputs['scm_score']
        scm_label = batch.scm_label
        batch_idx = batch.batch
        coords = batch.coords
        labels = batch.y

        for g in range(batch.num_graphs):
            mask = (batch_idx == g)
            if mask.sum() == 0:
                continue

            pred_g = scm_score[mask].cpu().numpy()
            true_g = scm_label[mask].cpu().numpy()
            coords_g = coords[mask].cpu().numpy()

            if labels[g].item() > 0 and true_g.sum() > 0:
                true_center = np.average(coords_g, axis=0, weights=true_g + 1e-6)
                pred_center = np.average(coords_g, axis=0, weights=pred_g + 1e-6)
                offset = np.linalg.norm(true_center - pred_center)
                center_offsets.append(offset)

    metrics = {
        'center_offset_mean': np.mean(center_offsets) if center_offsets else 0.0,
        'center_offset_std': np.std(center_offsets) if center_offsets else 0.0,
    }

    return metrics


def print_localization_metrics(tcp_metrics: Dict, scm_metrics: Dict) -> None:
    print("\n" + "=" * 60)
    print("TEMPORAL LOCALIZATION (TCP)")
    print("=" * 60)
    print(f"Peak Time Error  : {tcp_metrics['peak_time_error_mean']:.2f} +/- {tcp_metrics['peak_time_error_std']:.2f} (frames)")
    print(f"AUC (binary)     : {tcp_metrics.get('tcp_auc', np.nan):.4f}")

    print("\n" + "=" * 60)
    print("SPATIAL LOCALIZATION (SCM)")
    print("=" * 60)
    print(f"Center Offset    : {scm_metrics['center_offset_mean']:.4f} +/- {scm_metrics['center_offset_std']:.4f}")