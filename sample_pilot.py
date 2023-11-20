import os
import json 
from tqdm import tqdm
import random 
random.seed(2023)

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

    