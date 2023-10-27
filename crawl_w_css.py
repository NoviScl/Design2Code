from playwright.sync_api import sync_playwright
import requests
import re
import json
from tqdm import tqdm
import random 
random.seed(2023)

placeholder_image = "rick.jpg"

def replace_img_src(input_string):
    pattern = r'(<img[^>]*src=")[^"]*("[^>]*>)'
    replacement = r'\1' + placeholder_image + r'\2'
    return re.sub(pattern, replacement, input_string)

def fetch_and_embed_css(url, navigation_timeout=2000, request_timeout=10):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Navigate to the URL with timeout
        try:
            page.goto(url, timeout=navigation_timeout)
        except:
            return None

        # Extract stylesheets' hrefs using JavaScript
        stylesheets_hrefs = page.eval_on_selector_all("link[rel='stylesheet']", 'nodes => nodes.map(n => n.href)')

        inline_css = ""
        # Fetch and embed each external CSS
        content = page.content()
        for href in stylesheets_hrefs:
            try:
                response = requests.get(href, timeout=request_timeout)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
                css_content = response.text
                inline_css += '\n' + css_content + '\n\n'
            except:
                return None

        if "<style>" in content:
            content = content.replace("<style>", "<style>\n" + inline_css)
        else:
            content = content.replace('<head>', '<head>\n<style>' + inline_css + "</style>")
        
        content = replace_img_src(content)

        browser.close()

        return content

c4 = "../c4-validation.00000-of-00008.json"
urls = []
with open(c4, 'r') as f:
    for line in f:
        d = json.loads(line)
        urls.append(d["url"])

urls = list(set(urls))[:10]

counter = 1
for i, url in tqdm(enumerate(urls)):
    html_content = fetch_and_embed_css(url)
    if html_content:
        with open("c4-val-00000/{}.html".format(counter), "w", encoding="utf-8") as f:
            f.write(html_content)
        counter += 1
    

