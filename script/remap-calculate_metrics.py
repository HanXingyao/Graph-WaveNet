from script_ultils import *


if __name__ == '__main__':
    # ------------------------参数设置------------------------
    csv_file_path = 'exps/ja-8/ja-wave.csv'
    sample_threshold = 0.1 # 采样阈值, 0.1代表10%的概率进行反向采样

    calculate_mape_flag = False
    calculate_mae_rmse_flag = False
    calculate_accuracy_precision_recall_flag = False

    time_window_calculate_flag = True
    len_time_window = 13 # 时间窗长度

    half_k = len_time_window // 2

    # ------------------------读取数据------------------------
    # 原始数据    
    interval_length = 1
    path = os.path.dirname(__file__)
    json_path = os.path.join(path, '../data/JA_result.json')
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

    # ---------------------------------------------------
    pred_shape = pred_df.shape
    origin_real_df = origin_real_df.iloc[-pred_shape[0] + 100:, :]
    pred_df = pred_df.iloc[100:]
    origin_real_df = origin_real_df.reset_index(drop=True)
    pred_df = pred_df.reset_index(drop=True)
    print(origin_real_df.shape)
    print(pred_df.shape)
    # exit()

    pred_df = pred_df.applymap(lambda x: int(1) if x != 0 else 0)
    origin_real_df = origin_real_df.applymap(lambda x: int(1) if x != 0 else 0)

    # ------------------------计算MAPE-------------------------
    # if calculate_mape_flag:
        # mapes = calculate_mape_for_areas(origin_real_df, pred_df)
        # print('MAPE for each area: ')
        # for column, mape in mapes.items():
        #     print(f"{column}: {mape}")

    # ------------------------计算MAE、RMSE------------------------
    if calculate_mae_rmse_flag:
        maes = calculate_mae_for_areas(origin_real_df, pred_df)
        rmses = calculate_rmse_for_areas(origin_real_df, pred_df)

        print('#' * 10 + ' MAE for each area: ' + '#' * 10)
        for column, mae in maes.items():
            print(f"{column}: {mae}")

        print('#' * 10 + ' RMSE for each area: ' + '#' * 10)
        for column, rmse in rmses.items():
            print(f"{column}: {rmse}")   

    # ------------------------计算Accuracy、Precision、Recall------------------------
    if calculate_accuracy_precision_recall_flag:
        accuracys = calculate_accuracy_for_areas(origin_real_df, pred_df)
        precisions = calculate_precision_for_areas(origin_real_df, pred_df)
        recalls = calculate_recall_for_areas(origin_real_df, pred_df)

        # print(f'Accuracy for each area: {accuracys}')
        print('#' * 10 + ' Accuracy for each area: ' + '#' * 10)
        for column, accuracy in accuracys.items():
            print(f"{column}: {accuracy}")
        
        # print(f'Precision for each area: {precisions}')
        print('#' * 10 + ' Precision for each area: ' + '#' * 10)
        for column, precision in precisions.items():
            print(f"{column}: {precision}")

        # print(f'Recall for each area: {recalls}')
        print('#' * 10 + ' Recall for each area: ' + '#' * 10)
        for column, recall in recalls.items():
            print(f"{column}: {recall}")

        metrics = {
        'MAE': maes,
        'RMSE': rmses,
        'Accuracy': accuracys,
        'Precision': precisions,
        'Recall': recalls,
        }
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv('metrics.csv')
    # ------------------------时间窗估计预测指标------------------------
    if time_window_calculate_flag:
        k = len_time_window
        non_zero_counts_real = origin_real_df.apply(lambda x: (x == 1).sum())
        non_zero_counts_pred = pred_df.apply(lambda x: (x == 1).sum())
        # print("实际值中有任务的时间帧的个数:", non_zero_counts_real)
        # print("预测值中无任务的时间帧的个数:", non_zero_counts_pred)
        print("实际值中有任务的时间帧的个数:")
        for a in non_zero_counts_real.to_numpy():
            print(a)
        print("预测值中无任务的时间帧的个数:")
        for a in non_zero_counts_pred.to_numpy():
            print(a)
        evaluation_scores = pd.Series(dtype=float)
        for area in pred_df.columns:
            pred_indices = pred_df[area].index[pred_df[area] == 1]

            if len(pred_indices) == 0:
                evaluation_scores['area' + area[4:]] = 1.0
                continue

            correct_predictions = 0
            total_predictions = len(pred_indices)

            for index in pred_indices:
                start_index = max(index - half_k, 0)
                end_index = min(index + half_k, len(pred_df) - 1)

                if origin_real_df['real' + area[4:]].iloc[start_index:end_index + 1].any():
                    correct_predictions += 1
        
            evaluation_scores['area' + area[4:]] = round(correct_predictions / total_predictions * 100, 4)
        # print(evaluation_scores)
        for k, v in evaluation_scores.items():
            print(v)
# --------------------------------------------------------------------------------------------
    # flag =np.allclose(origin_real_df['real0'], pred_df['pred0'])
    # num_list = [0, 6, 9, 16]
    # flags = [origin_real_df[f'real{i}'].equals(pred_df[f'pred{i}']) for i in num_list]
    # print(flags)