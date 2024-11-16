import unittest
from unittest.mock import patch, MagicMock
from parts.bodys import Resnet  # 假设 Resnet 类在 your_module.py 中

class TestResnet(unittest.TestCase):

    @patch('parts.bodys.logger')
    @patch('parts.bodys.config')
    def test_resnet_init(self, mock_config, mock_logger):
        # 假设 config 模块有一个属性 config.some_value，模拟它的返回值
        mock_config.some_value = 'some_value'

        # 测试初始化 Resnet 时传入的 scale
        model = Resnet('18')

        # 断言 logger 是否记录了错误信息（当 scale 不在预期范围内）
        mock_logger.error.assert_called_with("scale not in ['18', '34', '50', '101', '152']")

        # 断言 Resnet 模型初始化是否正确
        self.assertTrue(hasattr(model, 'body'))
        # self.assertIsInstance(model.body, ResNet18_Weights)  # 假设 body 是 ResNet18_Weights 的实例

    def test_invalid_scale(self):
        # 测试传入无效 scale 时，是否会引发正确的错误
        with self.assertRaises(ValueError):  # 如果你在 Resnet 类中抛出 ValueError
            model = Resnet('999')  # 非法 scale

if __name__ == "__main__":
    unittest.main()
