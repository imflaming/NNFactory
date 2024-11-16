
import pandas as pd
import matplotlib.pyplot as plt
import random 

def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"#{r:02x}{g:02x}{b:02x}"

def plot(x , y_list, y_name_list , table_name ,xlabel = 'epoch' , save_path = 'date_plot.jpg'):
    for idx ,( y , n )in enumerate(zip(y_list, y_name_list)):
        plt.plot(x, y, label=n, color=random_color(),linestyle='--')

    plt.xticks(x, rotation=45)
    plt.legend()

    # 添加标题和标签
    plt.title(f'{table_name}')
    plt.xlabel(f'{xlabel}')
    plt.ylabel('null')

    plt.savefig(save_path, bbox_inches='tight')  # `bbox_inches='tight'` 可以去除边框空白区域
    plt.close()  # 关闭当前图形，释放内存
