
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict
import numpy as np
from sklearn.metrics import f1_score


def train_epoch(model: nn.Module, train_loader: DataLoader,
                criterion: nn.Module, optimizer: Optimizer,
                device: torch.device, epoch: int,
                clip_grad_norm: float = 5.0) -> Dict[str, float]:

    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_tcp_loss = 0.0
    total_scm_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in pbar:
        batch = batch.to(device)


        outputs = model(batch)
        loss_dict = criterion(outputs, batch)
        loss = loss_dict['total']


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()


        n_graphs = batch.num_graphs
        total_loss += loss.item() * n_graphs
        total_cls_loss += loss_dict['classification'].item() * n_graphs
        total_reg_loss += loss_dict['regularization'].item() * n_graphs
        total_tcp_loss += loss_dict['tcp'].item() * n_graphs
        total_scm_loss += loss_dict['scm'].item() * n_graphs

        preds = outputs['logits'].argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += n_graphs


        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls': f"{loss_dict['classification'].item():.4f}",
            'tcp': f"{loss_dict['tcp'].item():.4f}",
            'scm': f"{loss_dict['scm'].item():.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })

    return {
        'loss': total_loss / total,
        'cls_loss': total_cls_loss / total,
        'reg_loss': total_reg_loss / total,
        'tcp_loss': total_tcp_loss / total,
        'scm_loss': total_scm_loss / total,
        'accuracy': correct / total
    }


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader,
             criterion: nn.Module, device: torch.device,
             epoch: int) -> Dict[str, float]:

    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_tcp_loss = 0.0
    total_scm_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

    for batch in pbar:
        batch = batch.to(device)

        outputs = model(batch)
        loss_dict = criterion(outputs, batch)
        loss = loss_dict['total']

        n_graphs = batch.num_graphs
        total_loss += loss.item() * n_graphs
        total_cls_loss += loss_dict['classification'].item() * n_graphs
        total_reg_loss += loss_dict['regularization'].item() * n_graphs
        total_tcp_loss += loss_dict['tcp'].item() * n_graphs
        total_scm_loss += loss_dict['scm'].item() * n_graphs

        preds = outputs['logits'].argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += n_graphs

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })


    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'loss': total_loss / total,
        'cls_loss': total_cls_loss / total,
        'reg_loss': total_reg_loss / total,
        'tcp_loss': total_tcp_loss / total,
        'scm_loss': total_scm_loss / total,
        'accuracy': correct / total,
        'f1': f1
    }


def save_checkpoint(model: nn.Module, optimizer: Optimizer, epoch: int,
                    metrics: Dict[str, float], save_path: Path) -> None:

    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved: {save_path}")


def load_checkpoint(model: nn.Module, optimizer: Optimizer,
                    checkpoint_path: Path, device: torch.device) -> int:

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    logging.info(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    return epoch
