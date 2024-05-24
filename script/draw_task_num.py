import pandas as pd 
import matplotlib.pyplot as plt
import os
import numpy as np
# from draw_map import generate_random_colors

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
# df[5] = df[1]/df[0]

plt.figure()
bars = plt.bar(range(area_num), df.iloc[area_code, 0], color='skyblue')
plt.xticks(range(area_num), area_code)
plt.xlim(-1, area_num)
plt.ylim(0, 500)
plt.ylabel('Num of Task')
plt.title(f'[{map_name}]Task Num in each area')
# plt.show()

for bar in bars:
    height = bar.get_height()
    if height > 500:
        plt.text(bar.get_x() + bar.get_width() / 2, 300, str(height), ha='center', va='bottom')
    else:
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')
plt.savefig(save_path + '/task_num.png')
