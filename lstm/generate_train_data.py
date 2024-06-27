import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
import json, os


if __name__ == '__main__':
    pred_result_csv_path = 'tg-wave.csv'
    json_path = 'data/TG_result.json'

    pred_result_csv_path = os.path.join('./csv_result', pred_result_csv_path)
    pred_data = pd.read_csv(pred_result_csv_path)
    enhanced_data = pred_data['pred13']
    # enhanced_data = enhanced_data.apply(lambda x: 0 if x < 30 else x - 60)

    with open(json_path, 'r') as f:
        huawei_data = json.load(f)
        timestamp = list(huawei_data['0'].keys())[0]
        area_num = len(huawei_data['0'][timestamp]['start'])
        shape_y = len(huawei_data)
        real_index = ['real' + str(i) for i in range(0, area_num)]
        data = [
            [areas_data['start'][area] for j, area in enumerate(areas_data['start'])]
            for i, index in enumerate(huawei_data) if i != 0
            for timestep in list(huawei_data[index].keys())[:1]
            for areas_data in [huawei_data[index][timestep]]
        ]
        origin_real_df = pd.DataFrame(data, columns=real_index)
    origin_real_df = origin_real_df.applymap(lambda x: 1 if x != 0 else 0) # 映射到0-1

    enhanced_shape = enhanced_data.shape
    origin_real_df = origin_real_df.iloc[-enhanced_shape[0] + 50:, :]
    enhanced_data = enhanced_data.iloc[50:]
    origin_real_df = origin_real_df.reset_index(drop=True)
    enhanced_data = enhanced_data.reset_index(drop=True)

    origin_real_df = origin_real_df.iloc[:, 13] # 只取一个区域

    result = {
        'enhanced_data': enhanced_data,
        'original_data': origin_real_df
    }

    result = pd.DataFrame(result)
    result.to_csv(f'lstm/lstm_train_data.csv', index=False)
