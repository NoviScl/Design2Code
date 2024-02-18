from tqdm import tqdm
import numpy as np
import json
import os

reference_dir = "../../testset_full"
test_dirs = {
    "gpt4v_direct_prompting": "../../gpt4v_predictions_full/gpt4v_direct_prompting",
    "gpt4v_text_augmented_prompting": "../../gpt4v_predictions_full/gpt4v_text_augmented_prompting",
    "gpt4v_visual_revision_prompting": "../../gpt4v_predictions_full/gpt4v_visual_revision_prompting",
    "gemini_direct_prompting": "../../gemini_predictions_full/gemini_direct_prompting",
    "gemini_text_augmented_prompting": "../../gemini_predictions_full/gemini_text_augmented_prompting",
    "gemini_visual_revision_prompting": "../../gemini_predictions_full/gemini_visual_revision_prompting"
}
all_scores_dict = {}

# valid_files = [filename for filename in os.listdir(reference_dir) if filename.endswith(".png")]
# print ("total #egs: ", len(valid_files))

# ## check if the file is in all prediction directories
# for filename in os.listdir(reference_dir):
#     if filename.endswith(".html"):
#         if all([os.path.exists(os.path.join(test_dirs[key], filename.replace(".html", ".png"))) for key in test_dirs]):
#             file_name_list.append(filename)



# with open("all_scores_dict.json", "r") as f:
#     all_scores_dict = json.load(f)

# subset = os.listdir("/nlp/scr/clsi/Pix2Code/predictions_100/gpt4v_direct_prompting")
# print (subset, len(subset))

# for setting, scores in all_scores_dict.items():
#     print (setting)
#     matched_files = []
#     matched_scores = []
#     for k,v in scores.items():
#         if k in subset:
#             matched_files.append(k)
#             matched_scores.append(v)
#     print (len(matched_files))
#     print (np.mean(matched_scores, axis=0))

# print (matched_files)


# for test_dir in test_dirs:
#     print (test_dir)
#     print (len(os.listdir(test_dirs[test_dir])) // 2)

'''
number of predictions:

gpt4v_direct_prompting
776
gpt4v_text_augmented_prompting
775
gpt4v_visual_revision_prompting
755
gemini_direct_prompting
596
gemini_text_augmented_prompting
641
gemini_visual_revision_prompting
524
'''

'''
## merge missing gemini predictions into all_scores_dict
with open("all_scores_dict.json", "r") as f:
    all_scores_dict = json.load(f)

with open("prediction_file_name_list_gemini_missing.json", "r") as f:
    missing_gemini_files = json.load(f)

with open("res_dict_gemini_missing.json", "r") as f:
    res_dict_gemini_missing = json.load(f)


for k,v in res_dict_gemini_missing.items():
    print (k)
    for i in range(len(missing_gemini_files)):
        filename = missing_gemini_files[i]
        all_scores_dict[k][filename] = v[i]
    print (len(all_scores_dict[k]))

missing_scores = ["15915.html", "13579.html", "14423.html", "6941.html"]

## remove cases where the scores are all 0
for k,v in all_scores_dict.items():
    print (k)
    for filename in missing_scores:
        del v[filename]
    print (len(v))
    print (np.mean(list(v.values()), axis=0))

with open("all_scores_dict.json", "w") as f:
    json.dump(all_scores_dict, f, indent=4)
'''

with open("all_scores_dict.json", "r") as f:
    all_scores_dict = json.load(f)
files = list(all_scores_dict["gemini_direct_prompting"].keys())
all_scores_dict["gemini_ultra_direct_prompting"] = {}
all_scores_dict["gemini_ultra_text_augmented_prompting"] = {}
all_scores_dict["gemini_ultra_visual_revision_prompting"] = {}

with open("prediction_file_name_list_gemini_ultra.json", "r") as f:
    file_name_list = json.load(f)
with open("res_dict_gemini_ultra.json", "r") as f:
    res_dict = json.load(f)

for k,v in res_dict.items():
    k = k.replace("gemini_", "gemini_ultra_")
    print (k)
    for i in range(len(file_name_list)):
        filename = file_name_list[i]
        if filename in files:
            if sum(v[i]) > 0:
                all_scores_dict[k][filename] = v[i]
    print (len(all_scores_dict[k]))
    print (np.mean(list(all_scores_dict[k].values()), axis=0))
