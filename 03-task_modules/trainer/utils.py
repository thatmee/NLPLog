import random
import os
import numpy as np
import torch
from nlplog import FailurePredDataset, SequencePredDataset, Config, collate_for_sequence_pred


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_collate_and_bsz(dataset_builder, config:Config):
    if dataset_builder == FailurePredDataset:
        collate_fn = None
        batch_size = config.batch_size
    elif dataset_builder == SequencePredDataset:
        collate_fn = collate_for_sequence_pred
        batch_size = config.batch_size
    else:
        collate_fn = None
        batch_size = config.batch_size

    return collate_fn, batch_size


def get_lr_scheduler(optimizer, config:Config):
    if config.lr_scheduler == 'none':
        scheduler = None
    elif config.lr_scheduler == 'linear':
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=config.warmup_steps)
        scheduler_2 = torch.optim.lr_scheduler.LinearLR(optimizer, T_max=config.epochs-config.warmup_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler_2], milestones=[config.warmup_steps])
    elif config.lr_scheduler == 'reduceonplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)  # based on f1
    elif config.lr_scheduler == 'cosine':
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=config.warmup_steps)
        scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs - config.warmup_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler_2], milestones=[config.warmup_steps])

    return scheduler