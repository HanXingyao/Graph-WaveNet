import json, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import trange

def value_convert(value):
    if value >= 1:
        return 70
    else:
        return 0

if __name__ == '__main__':

    interval_length = 1
    every_k_days = 5

    path = os.path.dirname(__file__)
    file_path = os.path.join(path, '../data/TG_result.json')

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

    df = pd.DataFrame(huawei_dict).T
    df.index = pd.to_datetime(df.index)
    # print(df)
    for column in df.columns:
        df[column] = df[column].apply(lambda x: x[0])
        df[column] = df[column].apply(value_convert)
    
    # df.index = df.index.minute + df.index.hour * 60
    # df = df.T

    if not os.path.exists('every_{}_days'.format(str(every_k_days))):
        os.makedirs('every_{}_days'.format(str(every_k_days)))
    else:
        for file in os.listdir('every_{}_days'.format(str(every_k_days))):
            os.remove(os.path.join('every_{}_days'.format(str(every_k_days)), file))

    min_date = df.index.min().date()
    max_date = df.index.max().date()

    num_days = (max_date - min_date).days
    num_plots = num_days // every_k_days

    for i in trange(num_plots):
        start_date = min_date + pd.DateOffset(days=i*every_k_days)
        end_date = start_date + pd.DateOffset(days=every_k_days)
        # df_k_days = df[(df.index.date >= start_date) and (df.index.date < end_date)]
        start_date = pd.Timestamp(min_date + pd.DateOffset(days=i*every_k_days))
        end_date = pd.Timestamp(start_date + pd.DateOffset(days=every_k_days))
        df_k_days = df[(df.index >= start_date) & (df.index < end_date)]
        df_k_days.index = df_k_days.index.hour * 60 + df_k_days.index.minute
        df_k_days = df_k_days.T
        print(df_k_days.shape)

        plt.figure()
        _, ax = plt.subplots(figsize=(40, 2))
        sns.heatmap(df_k_days, cmap='YlGnBu', ax=ax, cbar=False, vmin=0, vmax=70)
        # plt.xlim(0, 575)
        # plt.xticks(np.arange(0, 575, 50), np.arange(0, 575, 50))
        # plt.yticks([i for i in range(0, 24)], [str(i + 1) for i in range(0, 24)])
        # plt.yticks(fontsize=4)

        # cbar = ax.figure.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.1)
        # cbar.ax.xaxis.set_ticks_position('top')
        # cbar.ax.xaxis.set_label_position('top')

        title = 'Huawei Task Density' + ' (every: ' + str(every_k_days) + ' days)'
        plt.xticks([])
        plt.xlabel('Time (minutes)')
        plt.ylabel('Areas')
        plt.title(title)
        # plt.subplots_adjust(left=0.3, right=0.8, top=0.98, bottom=0.1)

        pic_name = 'every_{}_days'.format(str(every_k_days)) + '/第' + str(i+1) + '张图.png'
        plt.savefig(pic_name)
        plt.close()

    # plt.figure()
    # # plt.axis('equal')
    # _, ax = plt.subplots(figsize=(60, 4))
    # sns.heatmap(df, cmap='YlGnBu', ax=ax, cbar=False)
    # plt.xlim(0, 575)
    # plt.xticks(np.arange(0, 575, 50),
    #             np.arange(0, 575, 50))
    # plt.yticks([i for i in range(0, 24)], [str(i + 1) for i in range(0, 24)])

    # plt.yticks(fontsize=4)

    # cbar = ax.figure.colorbar(ax.collections[0], ax=ax, orientation='horizontal',
    #                             pad=0.1)
    # cbar.ax.xaxis.set_ticks_position('top')
    # cbar.ax.xaxis.set_label_position('top')
    # # plt.show()
    # title = 'Huawei Task Density' + ' (Interval: ' + str(interval_length) + ' minutes)'

    # plt.xlabel('Time (minutes)')
    # plt.ylabel('Areas')
    # plt.title(title)
    # plt.subplots_adjust(left=0.3, right=0.8, top=0.98, bottom=0.1)
    # pic_name = 'pics/horizon_task_density_pics/horizon_huawei_task_density_' + str(interval_length) + '.png'
    # plt.savefig(pic_name)
    # plt.show()
