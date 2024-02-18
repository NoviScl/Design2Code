from tqdm import tqdm
import numpy as np
import json
import os
from bs4 import BeautifulSoup
import shutil

data_dirs = {
        "gemini_direct_prompting": "../../gemini_predictions_full/gemini_direct_prompting",
        "gemini_text_augmented_prompting": "../../gemini_predictions_full/gemini_text_augmented_prompting",
        "gemini_visual_revision_prompting": "../../gemini_predictions_full/gemini_visual_revision_prompting",
        "gpt4v_direct_prompting": "../../gpt4v_predictions_full/gpt4v_direct_prompting",
        "gpt4v_text_augmented_prompting": "../../gpt4v_predictions_full/gpt4v_text_augmented_prompting",
        "gpt4v_visual_revision_prompting": "../../gpt4v_predictions_full/gpt4v_visual_revision_prompting"
    }
final_file_list = "prediction_file_name_list_final.json"

def count_total_nodes(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return len(soup.find_all())

def avg_by_split():
    easy_split = []
    hard_split = []
    threshold = 133.5

    with open("all_scores_dict.json", "r") as f:
        all_scores_dict = json.load(f)
    filenames = list(all_scores_dict["gpt4v_direct_prompting"].keys())
    print ("#egs: ", len(filenames))

    data_dir = "../../testset_full"
    node_counts = []
    for filename in tqdm(filenames):
        with open(os.path.join(data_dir, filename)) as f:
            html_content = f.read()
        total_nodes = count_total_nodes(html_content)
        if total_nodes < threshold:
            easy_split.append(filename)
        else:
            hard_split.append(filename)

    print (len(easy_split))
    print (len(hard_split))

    for method , scores in all_scores_dict.items():
        print (method)
        easy_v = [scores[filename] for filename in easy_split]
        hard_v = [scores[filename] for filename in hard_split]
        print ("Easy Split: ")
        print ("N = ", len(easy_v))
        print ("mean: ", np.mean(easy_v, axis=0))

        print ("Hard Split: ")
        print ("N = ", len(hard_v))
        print ("mean: ", np.mean(hard_v, axis=0))

def collect_predictions():
    new_dir = "../../predictions_final/"
    ## remove the unused gemini and gpt4v predictions 
    with open(final_file_list, "r") as f:
        file_name_list_final = json.load(f)
    for method, data_dir in data_dirs.items():
        if "gemini" in method:
            subdir = "Gemini-Pro/"
        elif "gpt4v" in method:
            subdir = "GPT-4V/"

        new_dir_name = new_dir + subdir + method
        if not os.path.exists(new_dir_name):
            os.makedirs(new_dir_name)
        
        ## copy files over 
        for filename in file_name_list_final:
            shutil.copy(os.path.join(data_dir, filename), os.path.join(new_dir_name, filename))
            shutil.copy(os.path.join(data_dir, filename.replace(".html", ".png")), os.path.join(new_dir_name, filename.replace(".html", ".png")))

        print (new_dir_name, len(os.listdir(new_dir_name)))

# def sample():
#     ## sample 200 examples from each dataset 
#     data_dirs = {
#         "gemini_direct_prompting": "../../gemini_predictions_full/gemini_direct_prompting",
#     }
                
if __name__ == "__main__":
    collect_predictions()

