from tqdm import tqdm
import numpy as np
import json
import os
import shutil

## load the final set of files to be evaluated
with open("prediction_file_name_list_final.json", "r") as f:
    file_name_list_final = json.load(f)

## load gemini pro results
with open("prediction_file_name_list_gemini_public.json", "r") as f:
    file_name_list_gemini_public = json.load(f) 
with open("res_dict_gemini_public.json", "r") as f:
    res_dict_gemini_public = json.load(f)

## load gpt4v results
with open("prediction_file_name_list_gpt4v.json", "r") as f:
    file_name_list_gpt4v = json.load(f) 
with open("res_dict_gpt4v.json", "r") as f:
    res_dict_gpt4v = json.load(f)

scores_dict = {}
for k,v in res_dict_gemini_public.items():
    print (k)
    if k not in scores_dict:
        scores_dict[k] = {}
    for i in range(len(file_name_list_gemini_public)):
        filename = file_name_list_gemini_public[i]
        if filename in file_name_list_final:
            scores_dict[k][filename] = v[i]
    print (len(scores_dict[k]))
    print (np.mean(list(scores_dict[k].values()), axis=0))

for k,v in res_dict_gpt4v.items():
    print (k)
    if k not in scores_dict:
        scores_dict[k] = {}
    for i in range(len(file_name_list_gpt4v)):
        filename = file_name_list_gpt4v[i]
        if filename in file_name_list_final:
            scores_dict[k][filename] = v[i]
    print (len(scores_dict[k]))
    print (np.mean(list(scores_dict[k].values()), axis=0))

with open("all_scores_dict.json", "w") as f:
    json.dump(scores_dict, f, indent=4)

