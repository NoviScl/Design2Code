from crawl_w_css import *
from data_clean import *
from multiprocessing import Pool
import contextlib, joblib
from joblib import Parallel, delayed
from tqdm import tqdm


def filter_and_save(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        html_content = f.read()
    if html_content:
        html_content = html_content.strip()
        html_content_list = all_filters_train(html_content)
        if len(html_content_list) > 0:
            for html_content in html_content_list:
                html_content = html_content.strip()
                if len(html_content) > 0:
                    print(file_name)
                    print("passed")


file_names = [f"/nlp/scr/zyanzhe/pix2code_train/test_{i}_new.html" for i in range(1, 6)]
for file_name in file_names:
    filter_and_save(file_name)