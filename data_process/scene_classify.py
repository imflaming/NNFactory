from typing import Callable
from torch.utils.data import Dataset
import json
from logger import logger
from config import config

class SenceDataset(Dataset):
    def __init__(self, 
                 data_path: str ,
                 num_class : int = 10,
                 image_process: Callable = None,
                 label_process: Callable = None):
        
        logger.info(f"init dataset ...")
        logger.info(f"data_path : {data_path}")
        logger.info(f"num_class : {num_class}")
        logger.info(f"image_process : {image_process.__name__}")
        logger.info(f"label_process : {label_process.__name__}")
        self.numclasses = num_class
        self.image_process = image_process
        self.label_process = label_process

        self.labels = []
        self.image_paths = []
        with open( data_path, 'r', encoding= "utf-8") as data_f:
            lines = data_f.readlines()
            for line in lines:
                data = json.loads(line)
                image_path = data.get("image_path")
                label = data.get("label")
                self.image_paths.append(image_path)
                self.labels.append([label])
        logger.info(f"SenceDataset init over")
        logger.info(f"SenceDataset , datas count : {len(self.image_paths)} , labels num : {len(self.labels)}" )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.label_process(self.labels[idx], self.numclasses)
        input = self.image_process(self.image_paths[idx])
        return [input, label]