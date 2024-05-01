from script_ultils import *

if __name__ == '__main__':
    sample_threshold = 0.1 # 采样阈值, 0.1代表10%的概率进行反向采样
    csv_file_path = 'exps/域迁移/1模糊二分/map-wave.csv'
    # 取csv_file_path字符串中最后一个'/'及其前面的字符
    pic_name = 'Real vs Predicted Task Flow Data'
    pic_path = csv_file_path[0:csv_file_path.rfind('/')] + '/' + pic_name
    # print(pic_path)
    # ------------------------读取数据------------------------
    # 原始数据    
    interval_length = 1
    path = os.path.dirname(__file__)
    json_path = os.path.join(path, '../TG_result.json')
    with open(json_path, 'r') as f:
        huawei_data = json.load(f)
        real_index = ['real' + str(i) for i in range(0, 19)]
        data = [
            [areas_data['start'][area] for j, area in enumerate(areas_data['start'])]
            for i, index in enumerate(huawei_data) if i != 0
            for timestep in list(huawei_data[index].keys())[:1]
            for areas_data in [huawei_data[index][timestep]]
        ]
        origin_real_df = pd.DataFrame(data, columns=real_index)
    # origin_real_df = origin_real_df.iloc[int(origin_real_df.shape[0] * 0.8):, :] # 测试集
    origin_real_df = origin_real_df.applymap(lambda x: 1 if x != 0 else 0) # 映射到0-1
    # origin_real_df.to_csv('origin_real.csv', index=False)


    # 域迁移预测数据
    df = pd.read_csv(csv_file_path)

    columns = df.columns

    real_df = pd.DataFrame()
    pred_df = pd.DataFrame()

    def random_sample(value):
        if value > 40:
            if rd.uniform(0, 1) <= sample_threshold:
                return int(1)
            else:
                return int(0)
        else:
            return 0
        
    for column in columns:
        if 'real' in column:
            # real_df[column] = df[column].apply(convert_value)
            real_df[column] = df[column]
        if 'pred' in column:
            pred_df[column] = df[column].apply(random_sample)
            # pred_df[column] = df[column]

    real_df = origin_real_df
    real_df = real_df.iloc[-pred_df.shape[0]:, :]
    # ------------------------可视化(测试集与预测结果对比)------------------------
    draw_combined_matrix(real_df, pred_df, pic_path)
