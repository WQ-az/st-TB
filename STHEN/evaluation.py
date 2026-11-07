
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from src.models.full_model import SpatiotemporalEnsembleGNN
from src.data.dataset import BifurcationDataset, custom_collate
from src.utils.config import load_config

OUT_DIR = BASE_DIR / "results" / "figures" / "evaluation1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ['Null', 'Pitchfork', 'Fold', 'Transcritical', 'Period_doubling', 'Hopf']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def load_model_and_data(config_path, checkpoint_path, data_dir):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = SpatiotemporalEnsembleGNN(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    keys = list(state_dict.keys())
    for key in keys:
        if 'lin_src.weight' in key:
            new_key = key.replace('lin_src.weight', 'lin.weight')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        elif 'lin_dst.weight' in key:
            del state_dict[key]

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    test_dataset = BifurcationDataset(
        data_path=os.path.join(data_dir, 'test.npz'),
        normalize=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data'].get('batch_size', 32),
        shuffle=False,
        collate_fn=custom_collate
    )
    print(f"Test set size: {len(test_dataset)}")
    return model, test_loader, device

def plot_roc_curves(all_labels, all_probs, save_path):
    labels_binarized = label_binarize(all_labels, classes=range(6))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i in range(6):
        fpr, tpr, _ = roc_curve(labels_binarized[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], lw=2,
                label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Per-Class ROC Curves (One-vs-Rest)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.svg'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

def collect_localization_metrics(model, test_loader, device):
    model.eval()
    spatial_metrics = []
    temporal_metrics = []
    print("Collecting localization metrics per sample...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Localization"):
            batch = batch.to(device)
            outputs = model(batch)
            B = batch.y.shape[0]
            tcp_score = outputs['tcp_score']
            scm_score = outputs['scm_score']
            batch_idx = batch.batch
            coords = batch.coords
            labels = batch.y

            if hasattr(batch, 'tcp_label'):
                tcp_label = batch.tcp_label
                T = tcp_label.shape[1]
                pred_tcp = torch.zeros(B, T, device=device)
                for g in range(B):
                    mask = (batch_idx == g)
                    if mask.sum() > 0:
                        pred_tcp[g] = tcp_score[mask].mean(dim=0)
                for g in range(B):
                    if labels[g].item() == 0:
                        continue
                    true_peak = tcp_label[g].argmax().item()
                    pred_peak = pred_tcp[g].argmax().item()
                    peak_error = abs(pred_peak - true_peak)
                    temporal_metrics.append({
                        'true_label': labels[g].item(),
                        'peak_time_error': peak_error
                    })

            if hasattr(batch, 'scm_label'):
                scm_label = batch.scm_label
                for g in range(B):
                    mask = (batch_idx == g)
                    if mask.sum() == 0:
                        continue
                    pred_g = scm_score[mask].cpu().numpy()
                    true_g = scm_label[mask].cpu().numpy()
                    coords_g = coords[mask].cpu().numpy()

                    co = np.nan
                    if labels[g].item() > 0 and true_g.sum() > 0:
                        true_center = np.average(coords_g, axis=0, weights=true_g + 1e-6)
                        pred_center = np.average(coords_g, axis=0, weights=pred_g + 1e-6)
                        co = np.linalg.norm(true_center - pred_center)
                    spatial_metrics.append({
                        'true_label': labels[g].item(),
                        'center_offset': co,
                    })
    return temporal_metrics, spatial_metrics

def plot_pte(temporal_metrics, save_path):
    if not temporal_metrics:
        return
    pte = [m['peak_time_error'] for m in temporal_metrics if not np.isnan(m['peak_time_error'])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(pte, bins=20, color='#1f77b4', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(pte), color='red', linestyle='--', lw=2,
               label=f'Mean: {np.mean(pte):.2f} frames')
    ax.set_xlabel('Peak Time Error (frames)')
    ax.set_ylabel('Frequency')
    ax.set_title('(a) Peak Time Error Distribution')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    class_pte = {name: [] for name in CLASS_NAMES}
    for m in temporal_metrics:
        if not np.isnan(m['peak_time_error']):
            class_pte[CLASS_NAMES[m['true_label']]].append(m['peak_time_error'])
    data_to_plot = [class_pte[name] for name in CLASS_NAMES[1:]]
    bp = ax.boxplot(data_to_plot, labels=CLASS_NAMES[1:], patch_artist=True)
    for patch, color in zip(bp['boxes'], COLORS[1:]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Peak Time Error (frames)')
    ax.set_title('(b) PTE by Bifurcation Type')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png','.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png','.svg'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

def plot_center_offset(spatial_metrics, save_path):
    if not spatial_metrics:
        return
    co_list = [m['center_offset'] for m in spatial_metrics if not np.isnan(m['center_offset'])]
    if not co_list:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(co_list, bins=25, color='#d62728', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(co_list), color='darkred', linestyle='--', lw=2,
               label=f'Mean: {np.mean(co_list):.2f} μm')
    ax.set_xlabel('Center Offset (μm)')
    ax.set_ylabel('Frequency')
    ax.set_title('(a) Center Offset Distribution')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    class_co = {name: [] for name in CLASS_NAMES}
    for m in spatial_metrics:
        if not np.isnan(m['center_offset']):
            class_co[CLASS_NAMES[m['true_label']]].append(m['center_offset'])
    data_to_plot = [class_co[name] for name in CLASS_NAMES[1:]]
    bp = ax.boxplot(data_to_plot, labels=CLASS_NAMES[1:], patch_artist=True)
    for patch, color in zip(bp['boxes'], COLORS[1:]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Center Offset (μm)')
    ax.set_title('(b) Center Offset by Type')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png','.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png','.svg'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

def main():
    print("=" * 60)
    print("STHEN Evaluation Visualization: ROC + PTE + Center Offset")
    print("=" * 60)
    config_path = BASE_DIR / 'config' / 'v3.yaml'
    checkpoint_path = BASE_DIR  / 'results' / 'stgnn_bifurcation_v3' / 'checkpoints/best_model.pth'
    data_dir = BASE_DIR / 'data' / 'synthetic'


    print("\nLoading model and test data...")
    model, test_loader, device = load_model_and_data(config_path, checkpoint_path, data_dir)


    print("\nCollecting classification predictions for ROC...")
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collect logits"):
            batch = batch.to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs['logits'], dim=1)
            preds = outputs['logits'].argmax(dim=1)
            all_labels.append(batch.y.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()


    print("\nPlotting per-class ROC curves...")
    plot_roc_curves(all_labels, all_probs, str(OUT_DIR / 'roc_curves.png'))


    print("\nCollecting spatiotemporal localization metrics...")
    temporal_metrics, spatial_metrics = collect_localization_metrics(model, test_loader, device)
    print(f"  Temporal samples: {len(temporal_metrics)}")
    print(f"  Spatial samples: {len(spatial_metrics)}")


    plot_pte(temporal_metrics, str(OUT_DIR / 'peak_time_error.png'))
    plot_center_offset(spatial_metrics, str(OUT_DIR / 'center_offset.png'))

    print("\n" + "=" * 60)
    print("All selected figures generated successfully!")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
