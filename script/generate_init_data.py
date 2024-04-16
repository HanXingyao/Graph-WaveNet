import json, os, sys
import pandas as pd
from datetime import datetime
 
'''
匹配节点ID和坐标并按照任务发布时间整合数据 生成 init_data.json
'''

if __name__ == '__main__':
    # input file_name
    # -----------------------------------------------------------------
    map_name = 'TG'
    file_name = 'TG_CMD_0103-0303.xlsx'
    # -----------------------------------------------------------------

    map_data_dir = os.path.join(map_name, 'data.json')
    file_path = os.path.join(map_name, file_name)

    map_data = json.load(open(map_data_dir, 'r', encoding='utf-8'))
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    pos2node_id = {}

    for node_id, node_info in map_data.items():
        pos = (float(node_info['xpos']), float(node_info['ypos']))
        pos2node_id[pos] = node_id

    # print(df.columns)
    # start_pos_ls = [(x, y) for x, y in zip(df['start_x'] / 1000, df['start_y'] / 1000)]
    # end_pos_ls = [(x, y) for x, y in zip(df['end_x'] / 1000, df['end_y'] / 1000)]

    start_pos_ls = [(int(x) if x.is_integer() else x, int(y) if y.is_integer() else y) 
                    for x, y in zip(df['start_x'] / 1000, df['start_y'] / 1000)]
    
    end_pos_ls = [(int(x) if x.is_integer() else x, int(y) if y.is_integer() else y) 
                  for x, y in zip(df['end_x'] / 1000, df['end_y'] / 1000)]
    
    df['creation_date'] = pd.to_datetime(df['creation_date'], format='%Y-%m-%d %H:%M:%S.%f')
    df['creation_date'] = df['creation_date'].dt.floor('T')
    df['creation_date'] = df['creation_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    result = {}
    count = 0
    for index, row in df.iterrows():
        date = row['creation_date']

        if date not in result:
            result[date] = {'start': [], 
                            'end': []}

        if start_pos_ls[index] not in pos2node_id or end_pos_ls[index] not in pos2node_id:
            # print('start_pos_ls[index] not in pos2node_id')
            count += 1
            continue
        result[date]['start'].append(pos2node_id[start_pos_ls[index]])
        result[date]['end'].append(pos2node_id[end_pos_ls[index]])


    with open(os.path.join(map_name, 'init_data.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

    print('init_data.json generated successfully!')
    
    print(f'{count} data points are not in the map data')
