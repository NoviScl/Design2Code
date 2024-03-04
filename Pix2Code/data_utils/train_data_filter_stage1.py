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

parser = argparse.ArgumentParser()
parser.add_argument('--partition', type=str, default='00', help='data partition')
parser.add_argument('--begin', type=int, default=0)
args = parser.parse_args()

urls = []
partition = args.partition.strip()
c4 = "/nlp/scr2/nlp/data/commoncrawl/c4/en/c4-train.000{}-of-01024.json".format(str(partition))
with open(c4, 'r') as f:
    for line in f:
        d = json.loads(line)
        urls.append(d["url"])

urls = list(set(urls))
print ("total #urls: ", len(urls))

if not os.path.exists("/juice2/scr2/nlp/pix2code/zyanzhe/c4-train-html-s1-part{}".format(partition)):
    os.makedirs("/juice2/scr2/nlp/pix2code/zyanzhe/c4-train-html-s1-part{}".format(partition))

def filter_s1_and_save(inputs):
    i, url = inputs
    html_content = fetch_and_embed_css(url)
    if html_content:
        html_content = html_content.strip()
        if len(html_content) > 0:
            try:
                with open("/juice2/scr2/nlp/pix2code/zyanzhe/c4-train-html-s1-part{}/{}.html".format(partition, i), "w") as f:
                    f.write(html_content)
            except:
                pass


input_list = [(i, url) for i, url in enumerate(urls)]

input_list = input_list[args.begin:]
print (f"Start from {args.begin}, remain #urls: ", len(input_list))

"""
with tqdm_joblib(tqdm(total=len(input_list))) as progress_bar:
    res = list(tqdm(Parallel(n_jobs=4)(delayed(filter_and_save)(inputs) for inputs in input_list), total=len(input_list)))
"""
with tqdm_joblib(tqdm(total=len(input_list))) as progress_bar:
    res = list(tqdm(Parallel(n_jobs=4)(delayed(filter_s1_and_save)(inputs) for inputs in input_list), total=len(input_list)))