import enum
import pandas as pd
from enum import Enum
from logger import logger
from config import config
from utils.seaborn_plot import get_target_distribution
class Seasons(Enum):
    Spring = "Spring"
    Summer = "Summer"
    Fall = "Fall"
    Winter = "Winter"

class Strength(Enum):
    Weak = 1
    Normal = 2
    Strong = 3

class Frequency(Enum):
    DOES_NOT_APPLY = 0  # 不适用
    RARELY = 1          # 很少
    OCCASIONALLY = 2    # 偶尔
    FREQUENTLY = 3      # 经常
    OFTEN = 4           # 通常
    ALWAYS = 5          # 总是

class DailyUsage(Enum):
    LESS_THAN_1H = 0     # 少于1小时/天
    AROUND_1H = 1        # 大约1小时/天
    AROUND_2HS = 2       # 大约2小时/天
    MORE_THAN_3HS = 3    # 超过3小时/天
class BodyFrame(Enum):
    SMALL = 1   # 小型骨架
    MEDIUM = 2  # 中型骨架
    LARGE = 3   # 大型骨架
class Is_fintness(Enum):
    Health = 1
    NeedImprovement = 0

class Activity_Level_num(Enum):
    VeryLight= 1
    Light= 2
    Moderate= 3
    Heavy= 4
    Exceptional= 5


class SeverityImpairmentIndex(Enum):
    NONE = 0      # 无影响
    MILD = 1      # 轻微
    MODERATE = 2  # 中度
    SEVERE = 3    # 严重

    @classmethod
    def describe(cls, value):
        """根据枚举值返回描述"""
        descriptions = {
            cls.NONE: "无影响 - 互联网使用对生活没有明显负面影响。",
            cls.MILD: "轻微 - 存在一定程度的互联网使用问题，但未对生活造成重大影响。",
            cls.MODERATE: "中度 - 互联网使用已对日常生活产生一定负面影响。",
            cls.SEVERE: "严重 - 互联网使用严重干扰了日常生活和正常功能。",
        }
        return descriptions.get(cls(value), "未知级别")


from enum import Enum

class ActigraphyField(Enum):
    ID = "id"                     # 患者标识符，对应 train/test.csv 文件中的 id 字段
    STEP = "step"                 # 序列中的时间步长（整数）
    X = "X"                       # 腕戴设备沿 X 轴记录的加速度值（单位：g）
    Y = "Y"                       # 腕戴设备沿 Y 轴记录的加速度值（单位：g）
    Z = "Z"                       # 腕戴设备沿 Z 轴记录的加速度值（单位：g）
    ENMO = "enmo"                 # 由 WristPy 软件包计算的欧几里得模减一（ENMO）
    ANGLEZ = "anglez"             # 手臂相对于水平面的角度-Z
    NON_WEAR_FLAG = "non_wear_flag" # 手表是否佩戴的标志（0：佩戴；1：未佩戴）
    LIGHT = "light"               # 环境光照强度（单位：lux）
    BATTERY_VOLTAGE = "battery_voltage" # 电池电压（单位：mV）
    TIME_OF_DAY = "time_of_day"   # 数据采样时间（格式：%H:%M:%S.%9f）
    WEEKDAY = "weekday"           # 星期几，整数表示（1为周一，7为周日）
    QUARTER = "quarter"           # 年份的季度，整数值1至4
    RELATIVE_DATE_PCIAT = "relative_date_PCIAT" # 自 PCIAT 测试执行以来的天数（整数）

    @classmethod
    def describe(cls, field):
        """根据字段返回描述信息"""
        descriptions = {
            cls.ID: "患者标识符，对应 train/test.csv 文件中的 id 字段。",
            cls.STEP: "序列中的时间步长（整数）。",
            cls.X: "腕戴设备沿 X 轴记录的加速度值（单位：g）。",
            cls.Y: "腕戴设备沿 Y 轴记录的加速度值（单位：g）。",
            cls.Z: "腕戴设备沿 Z 轴记录的加速度值（单位：g）。",
            cls.ENMO: "由 WristPy 软件包计算的欧几里得模减一（ENMO），负值取零。",
            cls.ANGLEZ: "手臂相对于水平面的角度-Z。",
            cls.NON_WEAR_FLAG: "手表是否佩戴的标志（0：佩戴；1：未佩戴）。",
            cls.LIGHT: "环境光照强度（单位：lux）。",
            cls.BATTERY_VOLTAGE: "电池电压（单位：mV）。",
            cls.TIME_OF_DAY: "数据采样时间（格式：%H:%M:%S.%9f）。",
            cls.WEEKDAY: "星期几，整数表示（1为周一，7为周日）。",
            cls.QUARTER: "年份的季度，整数值1至4。",
            cls.RELATIVE_DATE_PCIAT: "自 PCIAT 测试执行以来的天数（整数，负值表示测试前收集的数据）。"
        }
        return descriptions.get(field, "未知字段")

def read_parquet_file(file_path = r"D:\dataset\CMI\series_test.parquet\id=001f3379\part-0.parquet"):
    df = pd.read_parquet(file_path)
    return df

def read_data_dictionary(path):
    df = pd.read_csv(path)
    return df

def _read_data_set(path):
    df = pd.read_csv(path)
    return df

def processor_train_data_set(path):
    """
    对train 数据预处
    """
    df = _read_data_set(path)
    [rows, columns ] = df.shape
    logger.info(f"train-set rows: {rows}")
    logger.info(f"train-set columns: {columns}")
    for name in df.columns:
        series = df.loc[:,[name]].squeeze()
        nan_count = _count_na( series )
        nan_percentage = nan_count / rows * 100
        logger.info(f"train column nan_count {nan_percentage:.2f}%,  name {name}")


    return df

def processor_val_data_set(path):
    df = _read_data_set(path)
    [rows, columns ] = df.shape
    logger.info(f"train-set rows: {rows}")
    logger.info(f"train-set columns: {columns}")
    for name in df.columns:
        series = df.loc[:,[name]].squeeze()
        nan_count = _count_na( series )
        nan_percentage = nan_count / rows * 100
        logger.info(f"val column nan_count {nan_percentage:.2f}%,  name {name}")

    continus = []
    # get_target_distribution(df)
    return df

def _count_na(values: pd.Series):
    # logger.info(f"{type(values)} {values}")
    n = values.isna().sum()
    return n

# 示例用法
if __name__ == "__main__":
    import os
    root = "D:\dataset\CMI"
    dicitionary = "data_dictionary.csv"
    file_path = r"D:\dataset\CMI\series_test.parquet\id=001f3379\part-0.parquet"

    df = read_parquet_file(file_path)
    df_dictionary = read_data_dictionary(os.path.join(root , dicitionary))
    df_train = processor_train_data_set(path= os.path.join( root , "train.csv") )

    print(df_dictionary.head())
    print(df_train.head())
