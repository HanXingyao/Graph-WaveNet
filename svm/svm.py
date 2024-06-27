import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from joblib import dump, load
import json, os


if __name__ == '__main__':
    pred_result_csv_path = 'tg-wave_1.csv'
    json_path = 'data/TG_result.json'

    class_weights = {0: 1, 1: 10}
    param_grid = {
        'C': [0.1, 1, 5, 10],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['linear', 'poly', 'rbf']
    }

    pred_result_csv_path = os.path.join('./csv_result', pred_result_csv_path)
    pred_data = pd.read_csv(pred_result_csv_path)
    # original_data = data['real0']
    enhanced_data = pred_data['pred3']
    # enhanced_data = enhanced_data.apply(lambda x: 0 if x < 30 else x - 60)
    # enhanced_data.to_csv('1.csv', index=False)
    # exit()
    

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
    enhanced_data = enhanced_data.iloc[40:-10]

    origin_real_df = origin_real_df.reset_index(drop=True)
    enhanced_data = enhanced_data.reset_index(drop=True)

    origin_real_df = origin_real_df.iloc[:, 3] # 只取一个区域

    save_result = {
        'enhanced_data': enhanced_data,
        'original_data': origin_real_df
    }
    save_result = pd.DataFrame(save_result)
    save_result.to_csv('1.csv', index=False)
    # exit()

    # ---------------------------------------------------

    features = []
    labels = []

    for i in range(5, len(origin_real_df) - 5):
        feature = enhanced_data[i-5:i+6].to_list()
        label = origin_real_df[i]
        features.append(feature)
        labels.append(label)

    features_df = pd.DataFrame(features)
    labels_df = pd.Series(labels)

    class_weights = {0: 1, 1: 6}
    X_train, X_test, y_train, y_test = train_test_split(features_df, labels_df, test_size=0.2, random_state=42, shuffle=False)

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=5.0, gamma='scale', class_weight=class_weights)
    model.fit(X_train_scaled, y_train)

    # y_pred = model.predict(X_test_scaled)
    decision_scores = model.decision_function(X_test_scaled)

    threshold = 0.9

    y_pred_threshold = (decision_scores > threshold).astype(int)

    report = classification_report(y_test, y_pred_threshold)
    print(report)

    results = pd.DataFrame({
        'y_pred': y_pred_threshold,
        'y_test': y_test
    })  
    results.to_csv('svm_results.csv', index=False)

    # dump(model, 'pth_files/svm_model.joblib')

    # plt.scatter(range(len(y_pred)), y_pred, label='SVM Prediction', s=1)
    # plt.scatter(range(len(y_test)), y_test, label='Original Label', s=1)
    # plt.xlabel('Sample Index')
    # plt.ylabel('Label')
    # plt.legend()
    # plt.show()

    # grid_search = GridSearchCV(SVC(class_weight=class_weights), param_grid, cv=5)
    # grid_search.fit(X_train_scaled, y_train)
    # print(grid_search.best_params_)
