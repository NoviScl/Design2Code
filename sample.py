from bs4 import BeautifulSoup
from collections import Counter
from tqdm import tqdm 
import os
import json
import pandas as pd
import numpy as np

def count_unique_tags(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    tags = [tag.name for tag in soup.find_all()]
    unique_tags = set(tags)
    return len(unique_tags)

def calculate_dom_depth(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    def get_max_depth(element, depth):
        children = element.find_all(recursive=False)
        if not children:
            return depth
        return max(get_max_depth(child, depth + 1) for child in children)

    return get_max_depth(soup, 0)

def count_total_nodes(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return len(soup.find_all())

def compute_stats_and_bucketize(numbers, k):
    if not numbers or k <= 0:
        return "Invalid input"

    average = sum(numbers) / len(numbers)
    minimum = min(numbers)
    maximum = max(numbers)

    # Calculate the range for each bucket
    range_size = (maximum - minimum) / k
    buckets = [[] for _ in range(k)]

    # Function to determine the bucket index for a number
    def get_bucket_index(num):
        if num == maximum:
            return k - 1
        return int((num - minimum) / range_size)

    # Assign numbers to buckets
    for num in numbers:
        bucket_index = get_bucket_index(num)
        buckets[bucket_index].append(num)

    return average, minimum, maximum, buckets

def interval_buckets(min_value, max_value, k):
    """
    Split the range [min_value, max_value] into k equal-interval buckets.

    :param min_value: The minimum value of the range.
    :param max_value: The maximum value of the range.
    :param k: Number of buckets.
    :return: A list of tuples, each representing the min and max values of a bucket.
    """
    if k <= 0:
        raise ValueError("Number of buckets (k) must be positive.")

    interval = (max_value - min_value) / k
    buckets = []

    for i in range(k):
        bucket_min = min_value + i * interval
        # To ensure the last bucket ends exactly at max_value
        bucket_max = min_value + (i + 1) * interval if i < k - 1 else max_value
        buckets.append((bucket_min, bucket_max))

    return buckets

def size_buckets(data, k):
    """
    Split the data into k buckets such that each bucket has approximately the same number of items.

    :param data: List of numbers to be bucketed.
    :param k: Number of buckets.
    :return: A list of tuples, each representing the min and max values of a bucket.
    """
    if k <= 0:
        raise ValueError("Number of buckets (k) must be positive.")

    # Convert data to a pandas Series for ease of computation
    series = pd.Series(data)
    # Calculate quantiles
    quantiles = np.linspace(0, 1, k+1)
    quantile_ranges = series.quantile(quantiles).tolist()

    # Create buckets
    buckets = [(quantile_ranges[i], quantile_ranges[i+1]) for i in range(len(quantile_ranges)-1)]

    return buckets


unique_tags = []
dom_depth = []
total_nodes = []

easy = []
mid = []
hard = []


directory = "/juice2/scr2/nlp/pix2code/testset_filter_round1"
for i in tqdm(range(1, 5000)):
        filename = str(i) + ".html"
        full_path = os.path.join(directory, filename)
        with open(full_path, "r", encoding="utf-8") as f:
            html_content = f.read() 
        nodes = count_total_nodes(html_content)

        if nodes >= 5 and nodes < 112:
            easy.append(filename)
        elif nodes >= 112 and nodes <= 287:
            mid.append(filename)
        else:
            hard.append(filename)

print ("nodes:")
print ("easy: ", len(easy))
print ("mid: ", len(mid))
print ("hard: ", len(hard))

test_set_split = {
    "easy": easy,
    "mid": mid,
    "hard": hard
}

with open("test_set_split_nodes.json", "w+") as f:
    json.dump(test_set_split, f, indent=4)




easy = []
mid = []
hard = []


directory = "/juice2/scr2/nlp/pix2code/testset_filter_round1"
for i in tqdm(range(1, 5000)):
        filename = str(i) + ".html"
        full_path = os.path.join(directory, filename)
        with open(full_path, "r", encoding="utf-8") as f:
            html_content = f.read() 
        depth = calculate_dom_depth(html_content)

        if depth >= 3 and depth < 10:
            easy.append(filename)
        elif depth >= 10 and depth <= 14:
            mid.append(filename)
        else:
            hard.append(filename)

print ("depth:")
print ("easy: ", len(easy))
print ("mid: ", len(mid))
print ("hard: ", len(hard))

test_set_split = {
    "easy": easy,
    "mid": mid,
    "hard": hard
}

with open("test_set_split_depth.json", "w+") as f:
    json.dump(test_set_split, f, indent=4)




easy = []
mid = []
hard = []


directory = "/juice2/scr2/nlp/pix2code/testset_filter_round1"
for i in tqdm(range(1, 5000)):
        filename = str(i) + ".html"
        full_path = os.path.join(directory, filename)
        with open(full_path, "r", encoding="utf-8") as f:
            html_content = f.read() 
        tags = count_unique_tags(html_content)

        if tags >= 5 and tags < 18:
            easy.append(filename)
        elif tags >= 18 and tags <= 24:
            mid.append(filename)
        else:
            hard.append(filename)

print ("tags:")
print ("easy: ", len(easy))
print ("mid: ", len(mid))
print ("hard: ", len(hard))

test_set_split = {
    "easy": easy,
    "mid": mid,
    "hard": hard
}

with open("test_set_split_tags.json", "w+") as f:
    json.dump(test_set_split, f, indent=4)




# directory = "/juice2/scr2/nlp/pix2code/testset_filter_round1"
# # for filename in tqdm(os.listdir(directory)):
# #     if filename.endswith(".html"):
# for i in tqdm(range(1, 5000)):
#         filename = str(i) + ".html"
#         full_path = os.path.join(directory, filename)
#         with open(full_path, "r", encoding="utf-8") as f:
#             html_content = f.read() 
#         # tags = count_unique_tags(html_content)
#         # depth = calculate_dom_depth(html_content)
#         nodes = count_total_nodes(html_content)

#         if nodes >= 5 and nodes < 112:
#             easy.append(filename)
#         elif nodes >= 112 and nodes <= 287:
#             mid.append(filename)
#         else:
#             hard.append(filename)

#         # if (tags >= 5 and tags < 18) and (depth >= 3 and depth < 10) and (nodes >= 5 and nodes < 112):
#         #     easy.append(filename)
#         # elif (tags >= 18 and tags <= 24) and (depth >= 10 and depth <= 14) and (nodes >= 112 and nodes <= 287):
#         #     mid.append(filename)
#         # elif (tags > 24) and (depth > 14) and (nodes > 287):
#         #     hard.append(filename)

# #         unique_tags.append(tags)
# #         dom_depth.append(depth)
# #         total_nodes.append(nodes)
        
# # print ("tags: ", size_buckets(unique_tags, 3))
# # print ("depth: ", size_buckets(dom_depth, 3))
# # print ("nodes: ", size_buckets(total_nodes, 3))

# print ("easy: ", len(easy))
# print ("mid: ", len(mid))
# print ("hard: ", len(hard))

# test_set_split = {
#     "easy": easy,
#     "mid": mid,
#     "hard": hard
# }

# with open("test_set_split_nodes.json", "w+") as f:
#     json.dump(test_set_split, f, indent=4)

# a, mi, ma, b = compute_stats_and_bucketize(unique_tags, 5)
# print ("number of unique tags: avg: {}, min: {}, max: {}".format(a, mi, ma))
# a, mi, ma, b = compute_stats_and_bucketize(dom_depth, 5)
# print ("dom depth: avg: {}, min: {}, max: {}".format(a, mi, ma))
# a, mi, ma, b = compute_stats_and_bucketize(total_nodes, 5)
# print ("number of total nodes: avg: {}, min: {}, max: {}".format(a, mi, ma))
