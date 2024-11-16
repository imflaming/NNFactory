import torch
import torch.nn.functional as F
from logger import logger

def one_hot_cross_entropy_loss(predictions, targets, epsilon=1e-9):
    """
    计算基于独热编码标签的交叉熵损失。
    
    参数:
        predictions (torch.Tensor): 模型的输出预测值，形状为 (batch_size, num_classes)，需要经过 softmax 处理。
        targets (torch.Tensor): 独热编码的标签，形状为 (batch_size, num_classes)。
        epsilon (float): 防止数值不稳定的小值，用于避免 log(0)。
        
    返回:
        torch.Tensor:
    
    """
    logger.debug(f"loss func predictions: {predictions.shape}")
    logger.debug(f"loss func targets: {targets.shape}")
    
    # 对预测值加上 epsilon 防止 log(0)
    predictions = predictions.clamp(min=epsilon, max=1.0)
    logger.debug(f"loss targets device {targets.device} predictions.device {predictions.device}")
    # 计算交叉熵损失
    loss = -torch.sum(targets * torch.log(predictions), dim=1)
    logger.debug(f"loss.shape : {loss.shape}")
    
    mean_loss = loss.mean()
    logger.debug(f"mean_loss: {mean_loss}")
    # 返回批次的平均损失
    return loss.mean()
