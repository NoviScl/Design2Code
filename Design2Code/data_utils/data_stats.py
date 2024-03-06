from bs4 import BeautifulSoup
from collections import Counter
import os
import json
import pandas as pd
import numpy as np

from tqdm import tqdm 
from datasets import load_dataset

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
import matplotlib.pyplot as plt

html5_tags = [
    "!DOCTYPE", "a", "abbr", "address", "area", "article", "aside", "audio", "b", "base", "bdi", "bdo", "blockquote",
    "body", "br", "button", "canvas", "caption", "cite", "code", "col", "colgroup", "data", "datalist", "dd", "del",
    "details", "dfn", "dialog", "div", "dl", "dt", "em", "embed", "fieldset", "figcaption", "figure", "footer", "form",
    "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hr", "html", "i", "iframe", "img", "input", "ins", "kbd",
    "label", "legend", "li", "link", "main", "map", "mark", "meta", "meter", "nav", "noscript", "object", "ol",
    "optgroup", "option", "output", "p", "param", "picture", "pre", "progress", "q", "rp", "rt", "ruby", "s", "samp",
    "script", "section", "select", "small", "source", "span", "strong", "style", "sub", "summary", "sup", "svg", "table",
    "tbody", "td", "template", "textarea", "tfoot", "th", "thead", "time", "title", "tr", "track", "u", "ul", "var",
    "video", "wbr"
]


def count_unique_tags(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    tags = [tag.name for tag in soup.find_all()]
    unique_tags = set(tags)
    return len(unique_tags)

def update_tag_frequencies(html_content, tag_frequency_dict):
    soup = BeautifulSoup(html_content, 'html.parser')
    tags = [tag.name for tag in soup.find_all()]
    
    # Update the passed dictionary with counts from the current HTML content
    for tag in tags:
        if tag in tag_frequency_dict:
            tag_frequency_dict[tag] += 1
        else:
            tag_frequency_dict[tag] = 1
    return tag_frequency_dict

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

def pie_chart():
    topics = {
        "product": 4,
        "blog": 21,
        "company/org": 23,
        "homepage": 11,
        "news": 4,
        "forum": 3,
        "information": 9,
        # "login": 2,
        # "forms": 2,
        "others": 8
    }

    # Prepare data
    labels = topics.keys()
    sizes = topics.values()

    # Create pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.title('Percentage of Each Topic')
    plt.savefig('topic_pie_chart.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def websight():
    dataset = load_dataset("HuggingFaceM4/WebSight", cache_dir="/juice2/scr2/nlp/pix2code/huggingface")
    all_texts = [eg for eg in tqdm(dataset["train"]["text"])]

    all_lengths = []
    all_total_tags = []
    all_dom_depths = []
    all_unique_tags = []
    tag_frequencies = {}

    for html_content in tqdm(all_texts):
        try:
            html_content = html_content.strip()
            tag_frequencies = update_tag_frequencies(html_content, tag_frequencies)
            
            length = tokenizer(html_content)["input_ids"]
            all_lengths.append(len(length))

            total_tags = count_total_nodes(html_content)
            all_total_tags.append(total_tags)

            dom_depth = calculate_dom_depth(html_content)
            all_dom_depths.append(dom_depth)

            unique_tags = count_unique_tags(html_content)
            all_unique_tags.append(unique_tags)    
        except:
            continue 

    sorted_tag_frequency_dict = dict(sorted(tag_frequencies.items(), key=lambda item: item[1], reverse=True))
    filtered_tag_frequency_dict = {k: v for k, v in sorted_tag_frequency_dict.items() if k in html5_tags}
    # print ("tag frequency: ", filtered_tag_frequency_dict)
    print ("mean length: ", np.mean(all_lengths))
    print ("mean total tags: ", np.mean(all_total_tags))
    print ("mean dom depth: ", np.mean(all_dom_depths))
    print ("mean unique tags: ", np.mean(all_unique_tags))
    print ("tag type: ", len(filtered_tag_frequency_dict))

    with open("websight_stats.json", "w") as f:
        json.dump({
            "lengths": all_lengths,
            "total_tags": all_total_tags,
            "dom_depths": all_dom_depths,
            "unique_tags": all_unique_tags,
            "tag_frequencies": filtered_tag_frequency_dict
        }, f, indent=4)


def websight_stats():
    with open("websight_stats.json", "r") as f:
        stats = json.load(f)
    lengths = stats["lengths"]
    total_tags = stats["total_tags"]
    dom_depths = stats["dom_depths"]
    unique_tags = stats["unique_tags"]
    tag_frequencies = stats["tag_frequencies"]
    
    print ("mean length: ", np.mean(lengths), np.std(lengths))
    print ("mean total tags: ", np.mean(total_tags), np.std(total_tags))
    print ("mean dom depth: ", np.mean(dom_depths), np.std(dom_depths))
    print ("mean unique tags: ", np.mean(unique_tags), np.std(unique_tags))

if __name__ == "__main__":
    # websight()
    websight_stats()

    '''
    pie_chart() 

    all_counts = []
    tag_frequencies = {}
    all_filenames = []
    for filename in tqdm(os.listdir("../../testset_final")):
        if ".html" in filename:
            all_filenames.append(filename)
            full_path = os.path.join("../../testset_final", filename)
            with open(full_path, "r") as f:
                html_content = f.read() 

                tag_frequencies = update_tag_frequencies(html_content, tag_frequencies)
                
                # length = tokenizer(html_content)["input_ids"]
                # all_counts.append(len(length))

                # total_tags = count_total_nodes(html_content)
                # all_counts.append(total_tags)

                # dom_depth = calculate_dom_depth(html_content)
                # all_counts.append(dom_depth)

    #             unique_tags = count_unique_tags(html_content)
    #             all_counts.append(unique_tags)

    # print (len(all_counts))
    # print (np.mean(all_counts))
    # print (min(all_counts), max(all_counts))
    # print (np.std(all_counts))

    # ## get different percentiles 
    # numbers = np.array(all_counts)

    # # Find the indices corresponding to the 0th, 25th, 50th, 75th, and 100th percentiles
    # percentiles = [0, 25, 50, 75, 100]
    # percentile_values = np.percentile(numbers, percentiles)

    # # Find the closest indices corresponding to these percentile values
    # indices = [np.abs(numbers - pv).argmin() for pv in percentile_values]

    # indices_dict = dict(zip(percentiles, indices))
    # for k,v in indices_dict.items():
    #     print(f"{k}th percentile: {all_filenames[v]}")
    #     print (f"Value: {all_counts[v]}")
    
    sorted_tag_frequency_dict = dict(sorted(tag_frequencies.items(), key=lambda item: item[1], reverse=True))
    print (sorted_tag_frequency_dict)
    print (len(sorted_tag_frequency_dict))

    ## filter out tags with only one occurrence 
    filtered_tag_frequency_dict = {k: v for k, v in sorted_tag_frequency_dict.items() if k in html5_tags}
    print (len(filtered_tag_frequency_dict))
    '''

