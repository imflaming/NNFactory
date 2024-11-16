import numpy as np
import torch 
from logger import  logger

# 4 个图片的形状最好一样

def RICP(images,target , beta =   0.3  ):
    """
        images: [4 , c , w , h]
        target: [4 , n ]

        混合后的图片
        patched_images : [1 , c , w , h]
        标签 c_ : [4 , n ]  
        W_ : [4 , i]
    """
    logger.debug(f"beta: {beta}")

    I_x, I_y = images.size()[2:]
    logger.debug(f"I_x: {I_x} , I_y: {I_y}")

    w = int(np.round(I_x * np.random.beta(beta, beta)))
    h = int(np.round(I_y * np.random.beta(beta, beta)))
    w_ =  [w, I_x - w, w, I_x - w]
    h_ =  [h, h, I_y - h, I_y - h]
    logger.debug(f"w_: {w_}")
    logger.debug(f"h_: {h_}")
    # select and crop four images
    cropped_images = {}
    c_ =  {} # 为标签
    W_ =  {} # 为拼接后权重
    for k in range(4):
        index = torch.randperm(images.size(0))
        x_k = np.random.randint(0, I_x - w_[k] + 1) # 目的是随机裁剪
        y_k = np.random.randint(0, I_y - h_[k] + 1)
        cropped_images[k]  = images[index][:,  :, x_k :x_k + w_[k], y_k:y_k + h_[k]]
        c_[k]  = target[index].cuda()
        W_[k]  = w_[k]  * h_[k]  /  (I_x * I_y)

    patched_images = torch.cat(
                (torch.cat((cropped_images[0], cropped_images[1]),  2),
                torch.cat((cropped_images[2], cropped_images[3]),  2)),
            3)
    patched_images = patched_images.cuda()
    logger.debug(f"patched_images shape : {patched_images.shape} , c_ : {c_} , W_ : {W_}")
    return patched_images , c_ , W_ 

def RICP_LOSS_and_ACC(target, y_head , patchs_weights ,loss_function , acc_discriminant ):
    logger.debug(f"target : {target} , y_head:{y_head} , patchs_weights: {patchs_weights}")

    loss = sum([patchs_weights[k]  * loss_function(y_head, target[k])  for k in range(4)]) # 仔细核对这个loss的计算
    acc = sum([patchs_weights[k]  * acc_discriminant(y_head, target[k])[0]  for k in range(4)])

    logger.debug(f"loss : {loss} , acc:{acc} ")
    return loss , acc

