import torch

import cv2
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import numpy as np
from logger import logger
from config import config, Status

def process_image(image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], target_size=128):
    """
    处理CV2图像为PyTorch tensor并进行归一化

    Args:
        image_path (str): 图像文件的路径
        mean (list): 归一化时每个通道的均值，默认为ImageNet的值
        std (list): 归一化时每个通道的标准差，默认为ImageNet的值

    Returns:
        tensor: 归一化后的PyTorch张量
    """
    
    logger.debug(f"image_path : {image_path}")
    logger.debug(f"mean: {mean}")
    logger.debug(f"std: {std}")

    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    logger.debug(f"Original image size: {w}x{h}")
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    square_image = np.full((target_size, target_size, 3), 125, dtype=np.uint8)

    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    square_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    if config.STATUS == Status.debug:
        cv2.imwrite('debug.jpg',square_image)
        
    square_image = cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB)

    image_pil = Image.fromarray(square_image)
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将PIL图像转为Tensor，并自动将像素值缩放至[0, 1]
        transforms.Normalize(mean=mean, std=std)  # 进行归一化处理
    ])

    image_tensor = transform(image_pil)

    logger.debug(f"image_tensor.shape : {image_tensor.shape}")

    return image_tensor
