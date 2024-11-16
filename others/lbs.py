import torch
import torch.nn as nn
from logger import logger
from config import config

class LSR(nn.Module):
    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction
        logger.info(f"LSR init over ")


    def _one_hot(self, labels, classes = config.CLASSES_NUM, value=1):
        """
        独热向量
        """
        logger.debug( f"classes : {classes}" )
        logger.debug( f"labels {labels}" )
        labels = torch.tensor(labels)
        one_hot = torch.zeros(len(labels), classes)
        labels = labels.view(len(labels),  -1)
        value_added = torch.Tensor(len(labels),  1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        if len(one_hot) == 1:
            one_hot = one_hot.squeeze(0)

        logger.debug( f"one_hot.shape {one_hot.shape}" )
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """
        标签平滑
        """
        logger.debug(f"smooth_factor: {smooth_factor} length {length}")
        one_hot = self._one_hot(target, length, value=1  - smooth_factor)
        one_hot += (smooth_factor / length)
        logger.debug(f"smoothed one_hot: {one_hot}")
        return one_hot.to(target.device)