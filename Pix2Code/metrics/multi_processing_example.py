# import sys,os
# sys.path.append("/nlp/scr/zyanzhe/Pix2Code")
from Pix2Code.metrics.visual_score import visual_eval_v3_multi
from multiprocessing import Pool
import contextlib, joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import json
import os

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def print_multi_score(multi_score):
    final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score = multi_score
    print()
    print("final_size_score", final_size_score)
    print("Matched Text Score", final_matched_text_score)
    print("Position Score", final_position_score)
    print("Text Color Score", final_text_color_score)
    print("CLIP Score", final_clip_score)
    print("--------------------------------\n")

debug = False

reference_dir = "../../testset_full"
test_dirs = {
    "gpt4v_direct_prompting": "../../gpt4v_predictions_full/gpt4v_direct_prompting",
    "gpt4v_text_augmented_prompting": "../../gpt4v_predictions_full/gpt4v_text_augmented_prompting",
    "gpt4v_visual_revision_prompting": "../../gpt4v_predictions_full/gpt4v_visual_revision_prompting",
    "gemini_direct_prompting": "../../gemini_predictions_full/gemini_direct_prompting",
    "gemini_text_augmented_prompting": "../../gemini_predictions_full/gemini_text_augmented_prompting",
    "gemini_visual_revision_prompting": "../../gemini_predictions_full/gemini_visual_revision_prompting"
}

# test_dirs = {
#     "gpt4v_direct_prompting": "../../predictions_100/gpt4v_direct_prompting",
#     "gpt4v_text_augmented_prompting": "../../predictions_100/gpt4v_text_augmented_prompting",
#     "gpt4v_visual_revision_prompting": "../../predictions_100/gpt4v_visual_revision_prompting",
# }

file_name_list = []

## check if the file is in all prediction directories
for filename in os.listdir(reference_dir):
    if filename.endswith(".html"):
        if all([os.path.exists(os.path.join(test_dirs[key], filename.replace(".html", ".png"))) for key in test_dirs]):
            file_name_list.append(filename)

# file_name_list = file_name_list[:10]

# file_name_list = ['9412.html', '15385.html', '11625.html', '10582.html', '6315.html', '8512.html', '13935.html', '1895.html', '11465.html', '5672.html', '13775.html', '10612.html', '4272.html', '2.html', '13692.html']

print ("total #egs: ", len(file_name_list))
with open("prediction_file_name_list.json", "w") as f:
    json.dump(file_name_list, f, indent=4)

input_lists = []
for filename in file_name_list:

    input_pred_list = [os.path.join(test_dirs[key], filename.replace(".html", ".png")) for key in test_dirs]
    original = os.path.join(reference_dir, filename.replace(".html", ".png"))

    input_list = [input_pred_list, original]
    input_lists.append(input_list)

with tqdm_joblib(tqdm(total=len(input_lists))) as progress_bar:
    return_score_lists = list(tqdm(Parallel(n_jobs=16)(delayed(visual_eval_v3_multi)(input_list, debug=debug) for input_list in input_lists), total=len(input_lists)))

## cache all scores 
with open("return_score_lists.json", "w") as f:
    json.dump(return_score_lists, f, indent=4)

res_dict = {}
for key in test_dirs:
    res_dict[key] = []

for i, filename in enumerate(file_name_list):
    idx = 0
    return_score_list = return_score_lists[i]
    if return_score_list:
        for key in test_dirs:
            matched, final_score, multi_score = return_score_list[idx]
            idx += 1
            current_score = [final_score] + [item for item in multi_score]
            res_dict[key].append(current_score)
    else:
        print (filename + " didn't get a score")
        for key in test_dirs:
            res_dict[key].append([0, 0, 0, 0, 0, 0])

## cache all scores 
with open("res_dict.json", "w") as f:
    json.dump(res_dict, f, indent=4)

for key in test_dirs:
    print(key)
    current_res = np.mean(np.array(res_dict[key]), axis=0)
    print(current_res)