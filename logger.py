import logging
from config import config
import pytz
from datetime import datetime

# 设置中国时区
china_tz = pytz.timezone('Asia/Shanghai')
# 创建一个 logger 实例
logger = logging.getLogger("my_logger")

# 设置日志级别
logger.setLevel(config.LOGGER_LEVEL)  # 设置最低日志级别为 DEBUG，这样会记录 DEBUG 及以上级别的日志

class ChinaTimeFormatter(logging.Formatter):
    def converter(self, timestamp):
        # 返回中国时区的当前时间
        dt = datetime.fromtimestamp(timestamp, china_tz)
        return dt.timetuple()
    
# 创建控制台处理器，并设置日志级别
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # 控制台输出最低级别设置为 INFO

# 创建一个日志格式器并将其添加到处理器
formatter = ChinaTimeFormatter('%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s: %(lineno)d | %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)


# 将处理器添加到 logger
logger.addHandler(console_handler)



