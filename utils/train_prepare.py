

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import os
from logger import logger

def setup(rank, world_size):
    """
        world_size: 表示节点数
    """
    logger.info(f"ddp set up rank:{rank}/{world_size}")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    logger.info(f"master addr:{'localhost:12355'}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # win系统使用
    torch.cuda.set_device(rank)
    logger.info(f"ddp set up rank:{rank}/{world_size} over")

def setup_win(rank, world_size):
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    logger.info(f"ddp set up rank:{rank}/{world_size}")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    logger.info(f"master addr:{'localhost:12355'}")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)  # win系统使用
    torch.cuda.set_device(rank)
    logger.info(f"ddp set up rank:{rank}/{world_size} over")

def cleanup():
    dist.destroy_process_group()