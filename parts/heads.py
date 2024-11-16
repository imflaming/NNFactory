
import os
from config import config
from logger import logger
from torch import nn
from logger import logger 
import torch

class SenceClassifyHead(nn.Module):
    def __init__( self , classes_num: int ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features= 512,out_features = classes_num),
            nn.Sigmoid()
        )
    
    def forward( self , input ):
        logger.debug(f"input shape : {input.shape}")
        input = input.squeeze(-1).squeeze(-1)
        output = self.fc(input)
        logger.debug(f"output shape : {output.shape}")
        return output