import argparse
from train import train , Trainner
import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DEBUG"
from logger import logger
from config import config
import os
import torch.multiprocessing as mp
import torch
from utils.train_prepare import setup, cleanup , setup_win
from config import config
from logger import logger



def main(rank: int, world_size: int):
    setup_win(rank, world_size)
    # train(rank, world_size)
    trainner = Trainner(rank , world_size)
    trainner.train()



if __name__ == "__main__":
    import argparse
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF" 
    logger.info(f"start train worker")
    # world_size = torch.cuda.device_count()
    world_size = config.WORLD_SIZE
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    
