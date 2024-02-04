from crawl_w_css import *
from data_clean import *
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


def filter_s2_and_save(html_file):
    try:
        html_name = html_file.split("/")[-1]
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        if html_content is not None:
            html_content = html_content.strip()
            html_content_list = all_filters_train(html_content)
            if len(html_content_list) > 0:
                for j, html_content in enumerate(html_content_list):
                    c_name = html_name.replace(".html", f"_{j}.html")
                    html_content = html_content.strip()
                    if len(html_content) > 0:
                        with open(f"/juice2/scr2/nlp/pix2code/zyanzhe/c4-train-html-s1-part00-v1.2/{c_name}", "w") as f:
                            f.write(html_content)
        else:
            print(html_file, "not valid")
    except:
        pass

input_list = os.listdir("/juice2/scr2/nlp/pix2code/zyanzhe/c4-train-html-s1-part00")
input_list = [os.path.join("/juice2/scr2/nlp/pix2code/zyanzhe/c4-train-html-s1-part00", item) for item in input_list]
print(input_list[:5])

with tqdm_joblib(tqdm(total=len(input_list))) as progress_bar:
    res = list(tqdm(Parallel(n_jobs=16)(delayed(filter_s2_and_save)(inputs) for inputs in input_list), total=len(input_list)))