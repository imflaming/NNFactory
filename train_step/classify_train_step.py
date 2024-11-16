from torch import nn
import torch
import torch.optim as optim
from utils.plot import plot
from typing import Callable
from torch.optim.optimizer import Optimizer
from config import config 
from logger import logger

def classify_step_train(model,input,label,loss_function:Callable,optimizer:Optimizer):
    logger.debug(f"step train input.shape : {input.shape}")
    logger.debug(f"step train label.shape : {label.shape}")
    
    optimizer.zero_grad()
    pred = model(input)
    loss = loss_function(pred , label)
    logger.debug(f"step train loss : {loss}")
    return loss

