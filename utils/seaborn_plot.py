import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 示例数据
# 假设 train_df 是您的原始数据，包含连续变量和目标变量 'sii'
data = {
    'Basic_Demos-Age': [25, 30, 35, 40, 22, 28, 33, 38],
    'Physical-BMI': [22.5, 24.0, 26.5, 23.0, 21.5, 25.0, 27.0, 22.0],
    'Physical-Height': [175, 180, 165, 170, 168, 178, 160, 172],
    'Physical-Weight': [70, 80, 72, 65, 62, 76, 68, 75],
    'FGC-FGC_GSND': [12, 15, 11, 14, 10, 13, 9, 16],
    'sii': [1, 0, 1, 0, 1, 0, 1, 0]  # 目标变量
}
train_df = pd.DataFrame(data)

# 定义连续变量列
continuous_columns = [
    'Basic_Demos-Age', 'Physical-BMI', 'Physical-Height', 'Physical-Weight', 'FGC-FGC_GSND'
]
def get_target_distribution( train_df,  continuous_columns, target_column , save_path = "target_distribution.png"):
    # 选择连续变量和目标变量 sii
    data_continuous = train_df[continuous_columns + [target_column]]

    # 删除目标变量 sii 为 NaN 的行（假设需要处理 NaN）
    data_continuous = data_continuous.dropna(subset=[target_column])

    # 将宽格式数据转换为长格式
    data_long = pd.melt(data_continuous, id_vars= target_column , var_name="variable", value_name="value")

    # 设置 FacetGrid，并按连续变量绘制 KDE 图
    g = sns.FacetGrid(data_long, col="variable", col_wrap=4, height=3.5, aspect=1.2, sharex=False, sharey=False)

    # 将 KDE 图映射到 FacetGrid 中
    g.map_dataframe(
        sns.kdeplot,
        x="value",
        hue= target_column,
        fill=True,
        common_norm=False,
        palette="Set2",
        alpha=0.4,
        linewidth=1.5
    )

    # 显示图表
    g.fig.tight_layout()
    plt.show()
    # plt.imsave()
    g.fig.savefig(save_path , dpi=300, bbox_inches="tight")


# get_target_distribution(train_df , continuous_columns , 'sii')