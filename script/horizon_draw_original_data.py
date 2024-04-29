import json, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def apply_value(value):
    if value >= 1:
        return 45
    else:
        return 20

interval_list = [1]

if __name__ == '__main__':
    for interval_length in interval_list:
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, '../TG_result.json')

        with open(file_path, 'r') as f:
            huawei_data = json.load(f)
            huawei_dict = {}

            temp_dict = {}
            for i, index in enumerate(huawei_data):
                if i == 0:
                    continue
                timestep = list(huawei_data[index].keys())[0]
                areas_data = huawei_data[index][timestep]
                for area in areas_data['start']:
                    start_num, end_num = areas_data['start'][area], areas_data['end'][area]
                    if area not in temp_dict:
                        temp_dict[area] = [start_num, end_num]
                    else:
                        temp_dict[area][0] += start_num
                        temp_dict[area][1] += end_num
                if i % interval_length == 0:
                    huawei_dict[timestep] = temp_dict
                    temp_dict = {}

            # start_num_dict = {}
            # for timestemp in huawei_dict:
            #     start_ls = huawei_dict[timestemp]
        df = pd.DataFrame(huawei_dict)
        df = df.iloc[ : , int(df.shape[1] * 0.8):]
        df = df.T
        df.index = pd.to_datetime(df.index)
        print(df.shape)
        # print(df)
        for column in df.columns:
            df[column] = df[column].apply(lambda x: x[0])
            df[column] = df[column].apply(apply_value)
            

        plt.figure()
        # plt.axis('equal')
        _, ax = plt.subplots(figsize=(100, 4))
        sns.heatmap(df.T, cmap='viridis', ax=ax, cbar=False, vmin=0, vmax=70)
        plt.title('Huawei Task Density')
        plt.xlabel('Time Step')
        plt.ylabel('Area ID')
        # plt.xticks(np.arange(0, df.shape[0], 50),
        #            np.arange(0, df.shape[0], 50))
        plt.xticks([])
        plt.yticks([i for i in range(0, df.shape[1])], [str(i + 1) for i in range(0, df.shape[1])])

        cbar = ax.figure.colorbar(ax.collections[0], ax=ax, orientation='horizontal',
                                    pad=0.1, fraction=0.02)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        # plt.show()
        title = 'Huawei Task Density' + ' (Interval: ' + str(interval_length) + ' minutes)'

        plt.xlabel('Time (minutes)')
        plt.ylabel('Areas')
        plt.title(title)
        plt.subplots_adjust()
        pic_name = 'pics/horizon_task_density_pics/horizon_huawei_task_density_' + str(interval_length) + '.png'
        plt.savefig(pic_name)
        # plt.show()
