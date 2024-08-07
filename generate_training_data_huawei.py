from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import json
import pandas as pd
import random
import copy


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    # data = np.expand_dims(df.values, axis=-1)
    data = np.array(df.values.tolist(), dtype=np.float32)
    # print(data.shape)
    # raise ValueError(12)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        # time_in_day = np.squeeze(time_in_day) # Squeeze dimensions of time_in_day
        feature_list.append(time_in_day)
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        # dow_tiled = np.squeeze(dow_tiled, axis=0) # Squeeze dimensions of dow_tiled
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    json_file_path = args.traffic_df_filename
    interval_length = args.interval
    ratio = args.data_ratio

    # 把 huawei_data 里的数据排到 huawei_dict
    with open(json_file_path, 'r', encoding='utf-8') as file:
        huawei_data = json.load(file)
    
    # ----------------------------------------------------------------------------
    ''' 
    # 若一个区域在某一帧不是零, 那么将其前5帧和后5帧的数据都设置为1(考虑两端不能越界的情况)
    temp_huawei_data = copy.deepcopy(huawei_data)
    # for i in range(int((1 - ratio) * len(huawei_data)), len(huawei_data)):
    for i in range(len(huawei_data)):
        index = str(i)
        timestep = list(huawei_data[index].keys())[0]
        areas_data = huawei_data[index][timestep]
        for area in areas_data['start']:
            start_num, _ = areas_data['start'][area], areas_data['end'][area]
            if start_num != 0:
                for j in range(1, 6):
                    if i - j >= 0:
                        timestep = list(huawei_data[str(i - j)].keys())[0]
                        temp_huawei_data[str(i - j)][timestep]['start'][area] = 1
                        # print("###########")
                        # huawei_data[i - j][timestep]['end'][area] = 1
                for j in range(1, 6):
                    if i + j < len(huawei_data):
                        timestep = list(huawei_data[str(i + j)].keys())[0]
                        temp_huawei_data[str(i + j)][timestep]['start'][area] = 1
                        # print("###########")
                        # huawei_data[i + j][timestep]['end'][area] = 1
    huawei_data = temp_huawei_data
    '''
    # 预处理: 任务流特征增强, 有任务的时间帧, 前5帧和后5帧的数据进行正态分布增强
    window_size = 5 # 增强窗口半大小
    std_dev = 10 # 标准差
    temp_huawei_data = copy.deepcopy(huawei_data)

    for i in range(int((1 - ratio) * len(huawei_data)), len(huawei_data)):
        index = str(i)
        timestep = list(huawei_data[index].keys())[0]
        areas_data = huawei_data[index][timestep]

        start_idx = max(0, i - window_size)
        end_idx = min(len(huawei_data), i + window_size + 1)
        time_steps = np.arange(start_idx, end_idx)

        for area in areas_data['start']:
            start_num, _ = areas_data['start'][area], areas_data['end'][area]
            if start_num != 0:
                normal_distribution = 2500 * np.exp(-((time_steps - i) ** 2) / (2 * std_dev ** 2)) / (np.sqrt(2 * np.pi) * std_dev) - 40
                for j in range(start_idx, end_idx):
                    if j < len(huawei_data):
                        timestamp = list(huawei_data[str(j)].keys())[0]
                        temp_huawei_data[str(j)][timestamp]['start'][area] = normal_distribution[j - start_idx]
                        # print(normal_distribution[j - start_idx])

    huawei_data = temp_huawei_data
                
    # ----------------------------------------------------------------------------
    huawei_dict = {}

    # temp_dict 用来存储每个时间段的数据
    temp_dict = {}
    for i in range(int((1 - ratio) * len(huawei_data)), len(huawei_data)):
        index = str(i)
        if i == 0:
            continue
        timestep = list(huawei_data[index].keys())[0]
        # print(type(timestep))
        areas_data = huawei_data[index][timestep]
        for area in areas_data['start']:
            start_num, _ = areas_data['start'][area], areas_data['end'][area]
            if area not in temp_dict:
                # temp_dict[area] = [start_num, end_num]
                temp_dict[area] = [start_num]
            else:
                temp_dict[area][0] += start_num
                # temp_dict[area][1] += end_num
        if i % interval_length == 0:
            # 有无任务的映射
            # for area in temp_dict:
            #     if temp_dict[area][0] == 0:
            #         temp_dict[area][0] = 8 + random.uniform(-8.0, 8.0)
            #     else:
            #         temp_dict[area][0] = 60 + random.uniform(-8.0, 8.0)

            # 无任务的+8.5，有任务的不动
            for area in temp_dict:
                if temp_dict[area][0] == 0:
                    temp_dict[area][0] = 7 + random.uniform(-7.0, 7.0)
                # else:
                #     temp_dict[area][0] += 20
                # print(temp_dict[area][0])

            huawei_dict[timestep] = temp_dict
            temp_dict = {}

    df = pd.DataFrame(huawei_dict).T
    df.index = pd.to_datetime(df.index)
    # df = df.applymap(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
    # df.to_csv('tg.csv', index=False, header=False)
    # exit()
    # print(df)
    # print(seq_length_x, seq_length_y)

    # ----------------------------------------------------------------------------

    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    print(x_offsets, y_offsets)
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    print('df.shape', df.shape)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=args.dow,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=1, help="Interval Num.",)
    parser.add_argument("--data_ratio", type=float, default=0.125, help="Data ratio backforwards, the number of days is 59.",)
    parser.add_argument("--output_dir", type=str, default="data/tg-task", help="Output directory.",)
    parser.add_argument("--traffic_df_filename", type=str, default="data/TG_result.json",
                        help="Raw traffic readings.",)
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true',)

    args = parser.parse_args()
    days = int(5 * args.data_ratio)
    args.output_dir = args.output_dir + "-{}".format(args.interval) + f'-{days}'
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y':
            exit()
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)

    # huawei_data = json.load(open(args.traffic_df_filename, 'r', encoding='utf-8'))
    # huawei_dict = {}
    # for i in range(len(huawei_data)):
    #     index = str(i)
    #     timestep = list(huawei_data[index].keys())[0] # 从2024-01-01 00:00:00开始的时间戳，类型为str，间隔为1min

    #     areas_data = huawei_data[index][timestep]
    #     for area in areas_data['start']: # 一个时间段内的所有区域，共有20个区域，每个区域有start这个特征值
    #         start_num, _ = areas_data['start'][area]
    #         huawei_dict[timestep][area] = [start_num]

    # df = pd.DataFrame(huawei_dict).T
    # df.index = pd.to_datetime(df.index) # 将索引转换为时间戳
