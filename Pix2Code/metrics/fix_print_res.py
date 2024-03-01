import json
import numpy as np
import math

def update_list(alist):
    new_list = []
    for sublist in alist:
        if math.isnan(sublist[0]):
            sublist = [0.2 * sublist[-1], 0, 0, 0, 0, sublist[-1]]
        new_list.append(sublist)
    return new_list

def update_dict(name_path, res_path, current_dict): 
    with open(name_path, 'r') as file:
        file_name = json.load(file)

    with open(res_path, 'r') as file:
        res_dict = json.load(file)
    for key in res_dict:
        c_list = res_dict[key]
        c_list = update_list(c_list)
        new_dict = {}
        assert len(c_list) == len(file_name)
        for i, name in enumerate(file_name):
            new_dict[name] = c_list[i]

        if key not in current_dict:
            current_dict[key] = new_dict
        else:
            print(key, "already read")
    
    return current_dict

current_dict = {}

name_path = 'prediction_file_name_list_part1_new.json'
res_path = 'res_dict_part1_new.json'
current_dict = update_dict(name_path, res_path, current_dict)

name_path = 'prediction_file_name_list_part2_new.json'
res_path = 'res_dict_part2_new.json'
current_dict = update_dict(name_path, res_path, current_dict)

for key in current_dict:
    print(key)
    np_res = []
    for file in current_dict[key]:
        np_res.append(current_dict[key][file])
    np_res = np.array(np_res)
    print(np_res.shape)
    res = np.mean(np_res, axis=0)
    print(' & '.join([f"{num*100:.1f}" for num in res[1:]]))