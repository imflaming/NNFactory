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
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights 
)

import os
from config import config

os.environ['TORCH_HOME'] = config.TORCH_HOME_PATH

# model_dir
# Step 1: Initialize model with the best available weights

weights = Swin_T_Weights.DEFAULT
model = swin_t(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()
