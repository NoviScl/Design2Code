import sys,os
sys.path.append("/nlp/scr/zyanzhe/Pix2Code")

from Pix2Code.prompting.gpt4v_utils import index_text_from_html
from multiprocessing import Pool
import contextlib, joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import json


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


def index_text(html_file):
    try:
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        marker_html_content, text_dict = index_text_from_html(html_content)

        with open(html_file.replace(".html", "_marker.html"), "w") as f:
            f.write(marker_html_content)

        with open(html_file.replace(".html", ".json"), 'w') as fp:
            json.dump(text_dict, fp)
    except:
        pass

"""
input_list = [item for item in os.listdir("/juice2/scr2/nlp/pix2code/zyanzhe/websight_file") if item.endswith(".html")]
input_list = [os.path.join("/juice2/scr2/nlp/pix2code/zyanzhe/websight_file", item) for item in input_list]

print(len(input_list), input_list[:5])

with tqdm_joblib(tqdm(total=len(input_list))) as progress_bar:
    res = list(tqdm(Parallel(n_jobs=16)(delayed(index_text)(inputs) for inputs in input_list), total=len(input_list)))

input_list = [item for item in os.listdir("/juice2/scr2/nlp/pix2code/zyanzhe/c4-train-html-s1-part00-v1.2") if item.endswith(".html")]
input_list = [os.path.join("/juice2/scr2/nlp/pix2code/zyanzhe/c4-train-html-s1-part00-v1.2", item) for item in input_list]

print(len(input_list), input_list[:5])

with tqdm_joblib(tqdm(total=len(input_list))) as progress_bar:
    res = list(tqdm(Parallel(n_jobs=16)(delayed(index_text)(inputs) for inputs in input_list), total=len(input_list)))
"""

input_list = [item for item in os.listdir("/juice2/scr2/nlp/pix2code/zyanzhe/websight_file_164k") if item.endswith(".html")]
input_list = [os.path.join("/juice2/scr2/nlp/pix2code/zyanzhe/websight_file_164k", item) for item in input_list]

print(len(input_list), input_list[:5])

with tqdm_joblib(tqdm(total=len(input_list))) as progress_bar:
    res = list(tqdm(Parallel(n_jobs=4)(delayed(index_text)(inputs) for inputs in input_list), total=len(input_list)))