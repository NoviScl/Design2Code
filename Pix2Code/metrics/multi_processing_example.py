import sys,os
sys.path.append("/nlp/scr/zyanzhe/Pix2Code")

from Pix2Code.metrics.visual_score import visual_eval_v3_multi
from multiprocessing import Pool
import contextlib, joblib
from joblib import Parallel, delayed
from tqdm import tqdm


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

debug = True

reference_dir = "../../testset_100"

test_dirs = {"websight": "../../predictions_100/websight",\
             "direct_prompting": "../../predictions_100/gpt4v_direct_prompting", \
             "text_augmented_prompting": "../../predictions_100/gpt4v_text_augmented_prompting", \
             "revision_prompting": "../../predictions_100/gpt4v_visual_revision_prompting"}

input_lists = []
for filename in ["16635.html", "8512.html", "13775.html", "13935.html"]:
    print(filename)

    input_pred_list = [os.path.join(test_dirs[key], filename.replace(".html", ".png")) for key in test_dirs]
    original = os.path.join(reference_dir, filename.replace(".html", ".png"))

    input_list = [input_pred_list, original]
    input_lists.append(input_list)

with tqdm_joblib(tqdm(total=len(input_lists))) as progress_bar:
    return_score_lists = list(tqdm(Parallel(n_jobs=4)(delayed(visual_eval_v3_multi)(input_list) for input_list in input_lists), total=len(input_lists)))

for i, filename in enumerate(["16635.html", "8512.html", "13775.html", "13935.html"]):
    print(filename)
    return_score_list = return_score_lists[i]
    idx = 0
    for key in test_dirs:
        matched, final_score, multi_score = return_score_list[idx]
        idx += 1
    
        print(f"{key} score: ", final_score)
        print_multi_score(multi_score)