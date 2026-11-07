
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import logging

from src.utils.config import (
    load_config, set_seed, get_device,
    create_exp_dirs, setup_logging,
    count_parameters, EarlyStopping
)
from src.data.dataset import build_dataloaders
from src.models.full_model import SpatiotemporalEnsembleGNN
from src.training.loss import TotalLoss
from src.training.trainer import train_epoch, validate, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Train Spatiotemporal Ensemble GNN')
    parser.add_argument('--config', type=str, default='config/v3.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    logger = setup_logging(
        log_level=config['logging']['level'],
        log_file=None
    )


    exp_dirs = create_exp_dirs(config)
    logger.info(f"Experiment: {config['experiment']['name']}")

    set_seed(config['seed'], config['reproducibility']['deterministic'])

    device = get_device(config['hardware']['device'], config['hardware']['gpu_id'])

    logger.info("Loading data...")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=config['paths']['data_dir'] + '/synthetic',
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples")
    logger.info(f"Val:   {len(val_loader.dataset)} samples")
    logger.info(f"Test:  {len(test_loader.dataset)} samples")

    logger.info("Building model...")
    model = SpatiotemporalEnsembleGNN(config).to(device)
    param_counts = count_parameters(model)
    logger.info(f"Model parameters: {param_counts['total']:,} "
                f"({param_counts['trainable']:,} trainable)")

    loss_config = config['loss']
    loc_config = loss_config.get('localization', {})
    criterion = TotalLoss(
        focal_alpha=loss_config['focal']['alpha'],
        focal_gamma=loss_config['focal']['gamma'],
        reg_lambda=loss_config['cell_influence']['lambda_reg'],
        reg_margin=loss_config['cell_influence']['margin'],
        lambda_tcp=loc_config.get('lambda_tcp', 0.5),
        lambda_scm=loc_config.get('lambda_scm', 0.5)
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['training']['betas'],
        eps=config['training']['eps']
    )

    warmup_epochs = config['training']['scheduler']['warmup_epochs']
    total_epochs = config['training']['num_epochs']

    if warmup_epochs > 0:

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=config['training']['scheduler']['eta_min']
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    else:

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=config['training']['scheduler']['eta_min']
        )


    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta'],
        mode='max' if 'accuracy' in config['training']['early_stopping']['monitor'] or 'f1' in config['training']['early_stopping']['monitor'] else 'min'
    )


    logger.info("Starting training...")

    monitor_metric = config['training']['early_stopping']['monitor']
    monitor_mode = 'max' if 'accuracy' in monitor_metric or 'f1' in monitor_metric else 'min'
    best_val_metric = float('-inf') if monitor_mode == 'max' else float('inf')

    for epoch in range(1, config['training']['num_epochs'] + 1):

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            clip_grad_norm=config['training']['clip_grad_norm']
        )

        val_metrics = validate(model, val_loader, criterion, device, epoch)


        current_lr = optimizer.param_groups[0]['lr']


        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_metrics['loss']:.4f} (cls={train_metrics['cls_loss']:.4f}, "
            f"tcp={train_metrics['tcp_loss']:.4f}, scm={train_metrics['scm_loss']:.4f}, "
            f"reg={train_metrics['reg_loss']:.4f}) | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} (cls={val_metrics['cls_loss']:.4f}, "
            f"tcp={val_metrics['tcp_loss']:.4f}, scm={val_metrics['scm_loss']:.4f}, "
            f"reg={val_metrics['reg_loss']:.4f}) | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )


        scheduler.step()


        if 'accuracy' in monitor_metric:
            current_metric = val_metrics['accuracy']
        elif 'f1' in monitor_metric:
            current_metric = val_metrics['f1']
        else:
            current_metric = val_metrics['loss']


        is_better = (current_metric > best_val_metric) if monitor_mode == 'max' else (current_metric < best_val_metric)
        if is_better:
            best_val_metric = current_metric
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                exp_dirs['checkpoints'] / 'best_model.pth'
            )

        if early_stopping(current_metric):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
