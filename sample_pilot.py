import os
import json
from screenshot import take_screenshot 
from sample_pilot import fetch_and_embed_css
from tqdm import tqdm
import os
import random 
random.seed(2023)

'''
with open("test_set_splits_all.json", "r") as f:
    labels = json.load(f)

## sample 40 examples from each category 
sampled = []
for k,v in labels.items():
    sampled.extend(random.sample(v, 40))

## copy files over to the new directory
for filename in tqdm(sampled):
    command = "cp /juice2/scr2/nlp/pix2code/testset_copy/{} /juice2/scr2/nlp/pix2code/pilot_testset/".format(filename)
    os.system(command)

    command = "cp /juice2/scr2/nlp/pix2code/testset_copy/{} /juice2/scr2/nlp/pix2code/pilot_testset/".format(filename.replace(".html", ".png"))
    os.system(command)
'''

## load original url dict 
with open("/juice2/scr2/nlp/pix2code/testset_filter_round1_url_dict.json", "r") as f:
    url_dict = json.load(f)

## take screenshots of the pilot test set data 
pilot_dir = "pilot_testset"
for filename in tqdm(os.listdir(pilot_dir)):
    if filename.endswith(".html") and "gpt4v" not in filename":
        url = url_dict[filename]
        html_content = fetch_and_embed_css(url)
        take_screenshot(os.path.join(pilot_dir, filename), os.path.join(pilot_dir, filename.replace(".html", ".png")))
    