from tqdm import tqdm
import numpy as np
import json
import os
from bs4 import BeautifulSoup

def count_total_nodes(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return len(soup.find_all())

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

# print ("mean: ", np.mean(node_counts))
# print ("median: ", np.median(node_counts))

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
    

