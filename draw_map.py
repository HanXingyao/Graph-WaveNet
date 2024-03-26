import json
import random
import matplotlib.pyplot as plt


def generate_random_color():
    """
    生成随机颜色的RGB值
    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r / 255, g / 255, b / 255  # 将RGB值归一化到[0, 1]范围


def generate_random_colors(num_colors):
    """
    生成指定数量的随机颜色列表
    """
    colors = [generate_random_color() for _ in range(num_colors)]
    return colors


# -----------------------------------------------------------------

task_data = json.load(open('data/huawei-task-seq.json', 'r', encoding='utf-8'))
task_area_name_ls = list(list(task_data['0'].values())[0]['start'].keys())
print('total_task_area_len:', len(task_area_name_ls))
print(task_area_name_ls)

# -----------------------------------------------------------------

area_id2name = json.load(open('name2id.json', 'r', encoding='utf-8'))
area_name2id = {v: k for k, v in area_id2name.items()}

# check 下哪些区域不在查询区域内
not_in_query_area_name_ls = []
for name in task_area_name_ls:
    if name not in area_name2id:
        not_in_query_area_name_ls.append(name)
print('not_in_query_area_name_ls:', len(not_in_query_area_name_ls))
if not_in_query_area_name_ls:
    print(not_in_query_area_name_ls)

# -----------------------------------------------------------------

map_data = json.load(open('data.json', 'r', encoding='utf-8'))
print(map_data.keys())

x_ls = []
y_ls = []
for node_id in map_data:
    # print(node_id, map_data[node_id])
    x_ls.append(float(map_data[node_id]['xpos']))
    y_ls.append(float(map_data[node_id]['ypos']))
print(x_ls)
print(y_ls)

# -----------------------------------------------------------------

no_areaCode_id_ls = []
area_code2node_id_ls = {}
for node_id in map_data:
    area_id = map_data[node_id]['value']
    if '@areaCode' not in area_id:
        area_id = int(area_id)
        if area_id not in no_areaCode_id_ls:
            no_areaCode_id_ls.append(area_id)
    else:
        area_id = int(area_id['@areaCode'])

    if area_id not in area_code2node_id_ls:
        area_code2node_id_ls[area_id] = []
    area_code2node_id_ls[area_id].append(node_id)
print('no_areaCode_id_ls:', no_areaCode_id_ls)

# -----------------------------------------------------------------

color_ls = generate_random_colors(len(area_code2node_id_ls))
plt.figure(figsize=(15, 10))

not_in_any_area_id_ls = []
not_in_task_area_id_ls = []
in_task_area_id_ls = []
for i, area_id in enumerate(area_code2node_id_ls):
    x_ls, y_ls = [], []
    if area_id2name.get(str(area_id)) is None:
        not_in_any_area_id_ls.append(area_id)
        continue
    elif area_id2name.get(str(area_id)) not in task_area_name_ls:
        not_in_task_area_id_ls.append(area_id)
        continue
    in_task_area_id_ls.append(area_id)

    for node_id in area_code2node_id_ls[area_id]:
        x = float(map_data[node_id]['xpos'])
        y = float(map_data[node_id]['ypos'])
        x_ls.append(x)
        y_ls.append(y)
    # print('area_id:', area_id, 'area_node_num:', len(x_ls))
    plt.scatter(x_ls, y_ls, c=[color_ls[i]], label=area_id)

print('not_in_any_area_id_ls:', len(not_in_any_area_id_ls), not_in_any_area_id_ls)
print('not_in_task_area_id_ls:', len(not_in_task_area_id_ls), not_in_task_area_id_ls)
print('in_task_area_id_ls:', len(in_task_area_id_ls), in_task_area_id_ls)
plt.axis('equal')
plt.show()
