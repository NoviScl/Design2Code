from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import requests
import re
import os
import json
import argparse
from tqdm import tqdm
import random 
random.seed(2023)

placeholder_image = "rick.jpg"

url_dict = {}

def replace_img_src(content):
    pattern = r'(<img[^>]*src=")[^"]*("[^>]*>)'
    replacement = r'\1' + placeholder_image + r'\2'
    return re.sub(pattern, replacement, content)

def fetch_and_embed_css(url, navigation_timeout=4000, request_timeout=10):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
    
            # Navigate to the URL with timeout
            page.goto(url, timeout=navigation_timeout)
    
            # Extract stylesheets' hrefs using JavaScript
            stylesheets_hrefs = page.eval_on_selector_all("link[rel='stylesheet']", 'nodes => nodes.map(n => n.href)')
    
            inline_css = ""
            # Fetch and embed each external CSS
            content = page.content()
            for href in stylesheets_hrefs:
                response = requests.get(href, timeout=request_timeout)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
                css_content = response.text
                inline_css += '\n' + css_content + '\n\n'
    
            if "<style>" in content:
                content = content.replace("<style>", "<style>\n" + inline_css)
            else:
                content = content.replace('<head>', '<head>\n<style>' + inline_css + "</style>")
            
            content = replace_img_src(content)
            browser.close()
    
            return content
    except:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--partition', type=str, required=True, help='data partition')
    args = parser.parse_args()

    # urls = []
    # partition = args.partition.strip()
    # c4 = "../c4/c4-validation.0000{}-of-00008.json".format(str(partition))
    # with open(c4, 'r') as f:
    #     for line in f:
    #         d = json.loads(line)
    #         urls.append(d["url"])

    # urls = list(set(urls))
    # print ("total #urls: ", len(urls))

    # if not os.path.exists("../c4-val-html-part{}".format(partition)):
    #     os.makedirs("../c4-val-html-part{}".format(partition))

    # counter = 1
    # for i, url in tqdm(enumerate(urls)):
    #     html_content = fetch_and_embed_css(url)
    #     if html_content:
    #         try:
    #             with open("../c4-val-html-part{}/{}.html".format(partition, counter), "w") as f:
    #                 f.write(html_content)
    #             url_dict["{}.html".format(counter)] = url
    #             counter += 1
    #         except:
    #             continue

    # with open("url_dict_part{}.json".format(partition), "w+") as f:
    #     json.dump(url_dict, f, indent=4)

    html_content = fetch_and_embed_css("https://candle.mpi-inf.mpg.de/")
    with open("../../testset_100/00.html", "w") as f:
        f.write(html_content)

