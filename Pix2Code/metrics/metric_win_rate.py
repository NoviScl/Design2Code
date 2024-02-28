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

name_path = 'prediction_file_name_list_part1.json'
res_path = 'res_dict_part1.json'
current_dict = update_dict(name_path, res_path, current_dict)

name_path = 'prediction_file_name_list_part2.json'
res_path = 'res_dict_part2.json'
current_dict = update_dict(name_path, res_path, current_dict)

tested_dict = {
    "1v2": "gpt4v_visual_revision_prompting",
    "2v2": "gpt4v_text_augmented_prompting",
    "3v2": "gpt4v_direct_prompting",
    "4v2": "gemini_text_augmented_prompting",
    "5v2": "gemini_visual_revision_prompting",
    "6v2": "pix2code_18b",
    "7v2": "websight",
}

for key in current_dict:
    print(key)

whole_res = []
for check_id in tested_dict:
    ann = []
    with open(f"/Users/zhangyanzhe/Downloads/{check_id}.txt", 'r') as file:
        for line in file:
            ann.append(line.strip())

    with open("/Users/zhangyanzhe/Downloads/sampled_for_annotation_v4/mapping.txt", 'r') as file:
        for line in file:
            chtml = line.split(":")[1].strip()[:-4] + ".html"
            tota_win = current_dict[tested_dict[check_id]][chtml][0] - current_dict["gemini_direct_prompting"][chtml][0]
            size_win = current_dict[tested_dict[check_id]][chtml][1] - current_dict["gemini_direct_prompting"][chtml][1]
            text_win = current_dict[tested_dict[check_id]][chtml][2] - current_dict["gemini_direct_prompting"][chtml][2]
            posi_win = current_dict[tested_dict[check_id]][chtml][3] - current_dict["gemini_direct_prompting"][chtml][3]
            colo_win = current_dict[tested_dict[check_id]][chtml][4] - current_dict["gemini_direct_prompting"][chtml][4]
            clip_win = current_dict[tested_dict[check_id]][chtml][5] - current_dict["gemini_direct_prompting"][chtml][5]

            if ann[int(line.split(":")[0].strip()[:-4]) - 1] == "tested":
                tgt = 1
            elif ann[int(line.split(":")[0].strip()[:-4]) - 1] == "baseline":
                tgt = 0
            elif ann[int(line.split(":")[0].strip()[:-4]) - 1] == "tie":
                tgt = -1
            else:
                raise NotImplementedError

            print(line.split(":")[0], tota_win, size_win, text_win, posi_win,colo_win, clip_win, ann[int(line.split(":")[0].strip()[:-4]) - 1])
            if tgt >= 0:
                whole_res.append([tota_win, size_win, text_win, posi_win, colo_win, clip_win, tgt])

whole_res = np.array(whole_res)
print(whole_res.shape)
np.save("whole_res.npy", whole_res)