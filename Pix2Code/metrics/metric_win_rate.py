import json

file_path = 'prediction_file_name_list_all.json'

with open(file_path, 'r') as file:
    file_name = json.load(file)

print(file_name)

file_path = 'res_dict_all.json'

with open(file_path, 'r') as file:
    res_dict = json.load(file)

for key in res_dict:
    print(key)

websight_win = 0
gemini_win = 0

with open("/Users/zhangyanzhe/Downloads/sampled_for_annotation_v4/mapping.txt", 'r') as file:
    for line in file:
        chtml = line.split(":")[1].strip()[:-4] + ".html"
        cindex = file_name.index(chtml)
        print(res_dict["websight"][cindex])
        print(res_dict["gemini_direct_prompting"][cindex])
        if res_dict["websight"][cindex] > res_dict["gemini_direct_prompting"][cindex]:
            websight_win += 1
        else:
            gemini_win += 1
        print(line.split(":")[0], gemini_win, websight_win)