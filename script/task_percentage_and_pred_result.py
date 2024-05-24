import pandas as pd 
import matplotlib.pyplot as plt
import os
import numpy as np
from draw_map import generate_random_colors

map_name = 'TG'
save_path = os.path.join('calculate_result', map_name)
if os.path.exists(save_path) is False:
    os.makedirs(save_path)
area_to_choose = {
    'BB': [0, 1, 2, 3, 4, 5],
    'JA': [0, 6, 7, 10, 13, 17, 18, 19],
    'TG': [3, 11, 12, 13, 15, 17]
}
area_code = area_to_choose[map_name]

csv_file = os.path.join('csv_result', map_name + '.csv')

df = pd.read_csv(csv_file, header=None)
# print(df)
area_num = len(area_code)
df[5] = df[1]/df[0]

plt.figure()
plt.bar(range(area_num), df.iloc[area_code, 5], color='skyblue')
plt.xticks(range(area_num), area_code)
plt.xlim(-1, area_num)
plt.ylim(0, 1)
plt.ylabel('Percentage of predicted task')
plt.title(f'[{map_name}]Percentage of task in each area')
# plt.show()
plt.savefig(save_path + '/task_percentage.png')

plt.figure()
colors = generate_random_colors(num_colors=3, seed=7)
plt.bar(np.arange(0, len(area_code)), df.iloc[area_code, 2], width=0.3, label='window len = 15', color=colors[0])
plt.bar(np.arange(0, len(area_code)) + 0.3, df.iloc[area_code, 3], width=0.3, label='window len = 10', color=colors[1])
plt.bar(np.arange(0, len(area_code)) + 0.6, df.iloc[area_code, 4], width=0.3, label='window len = 5', color=colors[2])
plt.axhline(y=60, color='r', linestyle='--')
plt.text(0, 60, '60%', color='r', va='bottom', ha='right')
plt.xticks(np.arange(0, len(area_code), step=1), area_code)
plt.ylim(0, 100)
plt.xlabel('area ID')
plt.ylabel('precision rate(%)')
plt.title(f'[{map_name}] Precision of Prediction')
plt.ylim(0, 100)
plt.legend()

# plt.show()
plt.savefig(save_path + '/precision.png')
