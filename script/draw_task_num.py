import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('task_num.csv', header=None)
print(df)
data_length = [59/5, 14/5, 11/5, 7/5]
# 画折线图
for i in range (0, df.shape[1], 2):
    if i + 1 < df.shape[1]:
        plt.figure()
        plt.plot(df.iloc[:, i], label='ground truth num')
        plt.plot(df.iloc[:, i + 1], label='prediction num')
        plt.xlabel('area ID')
        plt.ylabel('task num')
        plt.title(f'data length: {data_length[i//2]} days')
        plt.xticks(np.arange(0, len(df), step=1))
        plt.legend()
        # plt.show()
        plt.savefig(f'./task_num_{data_length[i//2]}.png')
        