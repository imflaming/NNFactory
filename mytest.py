import unittest
from unittest.mock import patch, MagicMock
from parts.bodys import Resnet  # 假设 Resnet 类在 your_module.py 中
import cv2 
import torch


img = cv2.imread("/home/hezhipeng/train_workspace/scene_trainworker/exam_images/000001.jpg")
img2 = cv2.imread("/home/hezhipeng/train_workspace/scene_trainworker/exam_images/000002.jpg")
img3 = cv2.imread("/home/hezhipeng/train_workspace/scene_trainworker/exam_images/000003.jpg")
img4 = cv2.imread("/home/hezhipeng/train_workspace/scene_trainworker/exam_images/000004.jpg")

net = Resnet('18')
img = cv2.resize(img,(256,256))
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
print(image_tensor.shape)
pred = net(image_tensor)
print( net )
w = net.state_dict()
print(w.keys())
print(pred.shape)


net.eval()

# 创建一个虚拟输入张量（通常是一个形状为 [batch_size, channels, height, width] 的 Tensor）
# 这里假设输入图像是 224x224 的 RGB 图像
dummy_input = torch.randn(1, 3, 512, 512)

# 定义保存的 ONNX 文件名
# onnx_file = 'resnet18_cut_model.onnx'

# # 使用 torch.onnx.export() 保存模型为 ONNX 格式
# torch.onnx.export(net,                # 模型
#                   dummy_input,          # 输入张量
#                   onnx_file,            # 保存路径
#                   export_params=True,   # 是否导出模型参数
#                   opset_version=12,     # ONNX 操作集版本
#                   do_constant_folding=True,  # 是否优化常量折叠
#                   input_names=['doudou_input'],    # 输入的名字
#                   output_names=['doudou_output'],  # 输出的名字
#                   dynamic_axes={'doudou_input': {0: 'batch_size'}, 'doudou_output': {0: 'batch_size'}})  # 支持动态 batch size
