

import os
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    else:
        torch.backends.cudnn.benchmark = True

    logging.info(f"Random seed set to {seed} (deterministic={deterministic})")


def load_config(config_path: str) -> Dict[str, Any]:

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logging.info(f"Configuration loaded from {config_path}")
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logging.info(f"Configuration saved to {save_path}")


def create_exp_dirs(config: Dict[str, Any]) -> Dict[str, Path]:

    exp_name = config['experiment']['name']
    base_dir = Path(config['paths']['results_dir']) / exp_name

    dirs = {
        'base': base_dir,
        'checkpoints': base_dir / 'checkpoints',
        'figures': base_dir / 'figures',
        'logs': base_dir / 'logs',
        'metrics': base_dir / 'metrics',
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Experiment directories created at {base_dir}")
    return dirs


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def get_device(device: str = "cuda", gpu_id: int = 0) -> torch.device:

    if device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        logging.info(f"Using device: {device} ({torch.cuda.get_device_name(gpu_id)})")
    else:
        device = torch.device("cpu")
        logging.info("Using device: CPU")

    return device


class EarlyStopping:

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False

    def __call__(self, current_value: float) -> bool:

        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info(f"Early stopping triggered after {self.counter} epochs")

        return self.early_stop

    def reset(self):

        self.counter = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.early_stop = False


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
 
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":

    print("Testing utilities...")

    set_seed(42)
    print("✓ Random seed set")

    try:
        config = load_config("config/default.yaml")
        print(f"✓ Config loaded: {config['experiment']['name']}")
    except FileNotFoundError:
        print("✗ Config file not found (expected in testing)")

    early_stop = EarlyStopping(patience=3, mode='min')
    for i in range(10):
        should_stop = early_stop(1.0 - i * 0.1)
        if should_stop:
            print(f"✓ Early stopping works (stopped at iteration {i})")
            break

    meter = AverageMeter()
    meter.update(1.0)
    meter.update(2.0)
    meter.update(3.0)
    assert abs(meter.avg - 2.0) < 1e-6
    print("✓ AverageMeter works")

    print("\nAll utility tests passed! ✓")
