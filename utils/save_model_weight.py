from config import config
from logger import logger
import torch
from torch import nn

def save_weight(model : nn.Module, save_path : str):
    try:
        logger.info(f"save model weight ...")
        w8 = model.state_dict()
        torch.save( w8 , save_path)
        logger.info(f"save model weight to path : {save_path}")
        return True
    except Exception as e:
        logger.error(f"error {e}")
        return False


def save_onnx(model : nn.Module, save_path : str , input_shape : list ):
    try:
        logger.info(f"save model onnx ...")
        dummy_input = torch.randn(input_shape)
        # 使用 torch.onnx.export() 保存模型为 ONNX 格式
        torch.onnx.export(model,                # 模型
                        dummy_input,          # 输入张量
                        save_path,            # 保存路径
                        export_params=True,   # 是否导出模型参数
                        opset_version=12,     # ONNX 操作集版本
                        do_constant_folding=True,  # 是否优化常量折叠
                        input_names=['doudou_input'],    # 输入的名字
                        output_names=['doudou_output'],  # 输出的名字
                        dynamic_axes={'doudou_input': {0: 'batch_size'}, 'doudou_output': {0: 'batch_size'}})  # 支持动态 batch size

        logger.info(f"save model onnx to path : {save_path}")
        return True
    except Exception as e:
        logger.error(f"error {e}")
        return False