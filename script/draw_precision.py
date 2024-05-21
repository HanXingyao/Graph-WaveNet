import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('pred_precision.csv', header=None)
print(df)
data_length = [59/5, 14/5, 11/5, 7/5]
rows_to_plot = [3, 11, 12, 13, 15, 17, 18]
# 画折线图
# for i in range (0, df.shape[1], 3):
#     if i + 2 < df.shape[1]:
#         plt.figure()
#         plt.plot(df.iloc[rows_to_plot, i], label='window len = 5')
#         plt.plot(df.iloc[rows_to_plot, i + 1], label='window len = 10')
#         plt.plot(df.iloc[rows_to_plot, i + 2], label='window len = 15')
#         plt.xlabel('area ID')
#         plt.ylabel('precision rate(%)')
#         plt.title(f'data length: {data_length[i//3]} days')
#         plt.xticks(np.arange(0, len(df), step=1))
#         plt.ylim(0, 100)
#         plt.legend()
#         # plt.show()
#         plt.savefig(f'D:\桌面\华为调度\CODE\Graph-WaveNet\pics/5.10/precision_{data_length[i//3]}.png')

# 画柱状图
for i in range (0, df.shape[1], 3):
    if i + 2 < df.shape[1]:
        plt.figure()
        plt.bar(np.arange(0, len(rows_to_plot)), df.iloc[rows_to_plot, i], width=0.3, label='window len = 5')
        plt.bar(np.arange(0, len(rows_to_plot)) + 0.3, df.iloc[rows_to_plot, i + 1], width=0.3, label='window len = 10')
        plt.bar(np.arange(0, len(rows_to_plot)) + 0.6, df.iloc[rows_to_plot, i + 2], width=0.3, label='window len = 15')
        plt.xlabel('area ID')
        plt.ylabel('precision rate(%)')
        plt.title(f'data length: {data_length[i//3]} days')
        # plt.xticks(np.arange(0, len(rows_to_plot), step=1))  # 设置横坐标为整数
        plt.xticks(np.arange(0, len(rows_to_plot), step=1), rows_to_plot)
        plt.ylim(0, 100)
        plt.legend()
        # plt.show()
        plt.savefig(f'D:\桌面\华为调度\CODE\Graph-WaveNet\pics/5.10/precision_{data_length[i//3]}.png')
