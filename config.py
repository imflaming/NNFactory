from enum import Enum
from typing import List
import logging

class Status(Enum):
    prod = 0
    train = 1
    eval = 2
    debug = 3

class Config:
    LOGGER_LEVEL = logging.INFO
    STATUS : Status = Status.train
    WORLD_SIZE : int = 1
    

    TORCH_HOME_PATH : str = '/home/hezhipeng/train_workspace/scene_trainworker/pretrain_model'
    BACKBORN_INPUT_SHAPE : List[int] = []
    BACKBORN_OUTPUT_SHAPE : List[int] = []
    CLASSIFICATION_HEAD_INPUT : List[int] = []
    CLASSES_NUM : int = 4

    DATA_PATH : str = "/home/hezhipeng/train_workspace/scene_trainworker/exam_images/exam_dataset/dataset.jsonl"
    TRAIN_BATCH_SIZE : int = 1

    TOTAL_EPOCHS : int = 100
    WARMUP_EPOCHS: int = 20

    MAX_TRAIN_LR : float = 0.01
    MIN_TRAIN_LR : float = 0.0001
    LR_PHASE : dict = {20: 0.005 , 50: 0.0025 , 75: 0.00125}


config = Config()