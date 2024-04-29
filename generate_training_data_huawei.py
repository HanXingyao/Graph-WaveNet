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

    # 把 huawei_data 里的数据排到 huawei_dict
    with open(json_file_path, 'r', encoding='utf-8') as file:
        huawei_data = json.load(file)
    
    # ----------------------------------------------------------------------------
    # 若一个区域在某一帧不是零，那么将其前5帧和后5帧的数据都设置为1(考虑两端不能越界的情况)
    # temp_huawei_data = copy.deepcopy(huawei_data)
    # for i in range(len(huawei_data)):
    #     index = str(i)
    #     timestep = list(huawei_data[index].keys())[0]
    #     areas_data = huawei_data[index][timestep]
    #     for area in areas_data['start']:
    #         start_num, _ = areas_data['start'][area], areas_data['end'][area]
    #         if start_num != 0:
    #             for j in range(1, 6):
    #                 if i - j >= 0:
    #                     timestep = list(huawei_data[str(i - j)].keys())[0]
    #                     temp_huawei_data[str(i - j)][timestep]['start'][area] = 1
    #                     # print("###########")
    #                     # huawei_data[i - j][timestep]['end'][area] = 1
    #             for j in range(1, 6):
    #                 if i + j < len(huawei_data):
    #                     timestep = list(huawei_data[str(i + j)].keys())[0]
    #                     temp_huawei_data[str(i + j)][timestep]['start'][area] = 1
                        # # print("###########")
                        # # huawei_data[i + j][timestep]['end'][area] = 1
    # huawei_data = temp_huawei_data
    # ----------------------------------------------------------------------------
    huawei_dict = {}
    
    # count = 0
    # for i in range(len(huawei_data)):
    #     index = str(i)
    #     timestep = list(huawei_data[index].keys())[0]
    #     areas_data = huawei_data[index][timestep]
    #     # area = 'ZD-D4-C14'
    #     area = 'ZD-D4-C16'
    #     start_num = areas_data['start'][area]
    #     if start_num != 0:
    #         count += 1
            # print(i, timestep, start_num)
    # print(count)
    # 开始迭代循环，temp_dict 用来存储每个时间段的数据
    temp_dict = {}
    for i, index in enumerate(huawei_data):
        if i == 0:
            continue
        timestep = list(huawei_data[index].keys())[0]
        # print(type(timestep))
        areas_data = huawei_data[index][timestep]
        for area in areas_data['start']:
            start_num, end_num = areas_data['start'][area], areas_data['end'][area]
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
            #         temp_dict[area][0] = 20 + random.uniform(-8.0, 8.0)
            #     else:
            #         temp_dict[area][0] = 60 + random.uniform(-8.0, 8.0)

            # for area in temp_dict:
            #     if temp_dict[area][0] != 0:
            #         temp_dict[area][0] = 100
            #     else:
            #         temp_dict[area][0] = 10

            huawei_dict[timestep] = temp_dict
            temp_dict = {}

    # 有任务就是1 无任务就是0
    # for timestep in huawei_dict:
    #     for area in huawei_dict[timestep]:
    #         if huawei_dict[timestep][area][0] != 0:
    #             huawei_dict[timestep][area][0] = 1
    #         if huawei_dict[timestep][area][1] != 0:
    #             huawei_dict[timestep][area][1] = 1
    # print(huawei_dict)

    df = pd.DataFrame(huawei_dict).T
    df.index = pd.to_datetime(df.index)
    # print(df)
    # print(seq_length_x, seq_length_y)

    # ----------------------------------------------------------------------------

    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    print(df.shape)
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
    parser.add_argument("--output_dir", type=str, default="data/tg-task", help="Output directory.",)
    parser.add_argument("--traffic_df_filename", type=str, default="data/TG_result.json",
                        help="Raw traffic readings.",)
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true',)

    args = parser.parse_args()
    args.output_dir = args.output_dir + "-{}".format(args.interval)
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y':
            exit()
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
