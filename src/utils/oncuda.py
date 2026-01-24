from __future__ import annotations

import os
import torch
import random
import numpy as np
from loguru import logger

DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pick_workers() -> int:
    num_workers = os.cpu_count() or 1
    return max(1, num_workers)


def set_seed(seed: int = 42) -> torch.device:
    device = DEVICE

    if device == torch.device("cuda:0"):
        torch.cuda.set_device(device)
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
    else:
        logger.error("CUDA is not available. Using CPU...")
        os._exit(1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return device


def setup_parallel() -> torch.device:
    device = set_seed()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(f"Parallel processing enabled on {torch.cuda.device_count()} GPUs")
    return device
