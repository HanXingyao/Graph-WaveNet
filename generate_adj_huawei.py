import json
import pickle
import numpy as np
import util


json_file_path = 'data/TG_result.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    huawei_data = json.load(file)

sensor_ids = []
for index in huawei_data:
    if sensor_ids:
        break
    timestep = list(huawei_data[index].keys())[0]
    areas_data = huawei_data[index][timestep]
    sensor_ids = list(areas_data['start'].keys())
print('Sensor ID list:', sensor_ids)

sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
print('Sensor ID dict:', sensor_id_to_ind)

adj_mx = np.diag(np.ones(len(sensor_ids)))
print('ADJ Matrix shape:', adj_mx.shape)

with open('data/sensor_graph/adj_mx_TG_new.pkl', 'wb') as f:
    pickle.dump((sensor_ids, sensor_id_to_ind, adj_mx), f)
print('#' * 100)

# --------------------------------------------------------------------------------------------

# sensor_ids, sensor_id_to_ind, adj_mx = util.load_pickle('data/sensor_graph/adj_mx.pkl')
# print(sensor_ids)
# print(sensor_id_to_ind)
# print(adj_mx.shape)
# # print(adj_mx)
# print('#' * 100)

# --------------------------------------------------------------------------------------------

sensor_ids, sensor_id_to_ind, adj_mx = util.load_pickle('data/sensor_graph/adj_mx_TG_new.pkl')
print(sensor_ids)
print(sensor_id_to_ind)
print(adj_mx.shape)
# print(adj_mx)
