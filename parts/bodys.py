from torchvision.io import decode_image

from torchvision.models import( 
    swin_t,Swin_T_Weights, 
    swin_s,Swin_S_Weights,
    swin_b,Swin_B_Weights,
    swin_v2_b, Swin_V2_B_Weights,
    swin_v2_s, Swin_V2_S_Weights,
    convnext_tiny , ConvNeXt_Tiny_Weights,
    convnext_small , ConvNeXt_Small_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_large , ConvNeXt_Large_Weights,
    resnet18,ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights 
    )

import os
from config import config
from logger import logger
from torch import nn

class Resnet(nn.Module):
    def __init__( self , scale: str , doudou_models : str = None):
        super().__init__()
        if scale not in ['18', '34', '50', '101', '152']:
            logger.error(f"scale not in ['18', '34', '50', '101', '152']")
        class_name = f"resnet{scale}"
        resnet_model = globals().get(class_name, None)()  # 获取类对象
        
        if resnet_model:
            self.doudou_resnet_body = nn.Sequential(
                resnet_model.conv1,
                resnet_model.bn1,
                resnet_model.relu,
                resnet_model.maxpool,
                resnet_model.layer1,
                resnet_model.layer2,
                resnet_model.layer3,
                resnet_model.layer4,
                resnet_model.avgpool
            )
        else:
            self.doudou_resnet_body = None

    def forward(self, input):
        logger.debug(f"resnet input shape : {input.shape}")
        out = self.doudou_resnet_body(input)
        logger.debug(f"resnet output shape : {out.shape}")
        return out
        
    

        


