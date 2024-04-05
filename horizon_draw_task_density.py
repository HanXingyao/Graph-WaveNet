import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':

    interval_length = 20

    with open('huawei-task-seq.json', 'r') as f:
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

        df = pd.DataFrame(huawei_dict).T
        df.index = pd.to_datetime(df.index)
        # print(df)
        for column in df.columns:
            df[column] = df[column].apply(lambda x: x[0])
        
        df.index = df.index.minute + df.index.hour * 60

        df = df.T

        plt.figure()
        # plt.axis('equal')
        _, ax = plt.subplots(figsize=(60, 4))
        sns.heatmap(df, cmap='YlGnBu', ax=ax, cbar=False)
        plt.xlim(0, 575)
        plt.xticks(np.arange(0, 575, 50),
                   np.arange(0, 575, 50))
        plt.yticks([i for i in range(0, 24)], [str(i + 1) for i in range(0, 24)])

        plt.yticks(fontsize=4)

        cbar = ax.figure.colorbar(ax.collections[0], ax=ax, orientation='horizontal',
                                  pad=0.1)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        # plt.show()
        title = 'Huawei Task Density' + ' (Interval: ' + str(interval_length) + ' minutes)'

        plt.xlabel('Time (minutes)')
        plt.ylabel('Areas')
        plt.title(title)
        plt.subplots_adjust(left=0.3, right=0.8, top=0.98, bottom=0.1)
        pic_name = 'pics/horizon_task_density_pics/horizon_huawei_task_density_' + str(interval_length) + '.png'
        plt.savefig(pic_name)