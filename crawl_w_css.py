from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import requests
import re
import json
from tqdm import tqdm
import random 
random.seed(2023)

placeholder_image = "rick.jpg"

def replace_img_src(content):
    pattern = r'(<img[^>]*src=")[^"]*("[^>]*>)'
    replacement = r'\1' + placeholder_image + r'\2'
    return re.sub(pattern, replacement, content)

# def remove_script(html_content):
#     while "<script" in html_content:
#         start_index = html_content.find("<script")
#         end_index = html_content.find("</script>") + 9  # +9 to account for the length of "</script>"

#         # Ensure both opening and closing tags are found
#         if start_index == -1 or end_index < 9:
#             break

#         # Remove content between and including the script tags
#         html_content = html_content[:start_index] + html_content[end_index:]

#     return html_content
        
# def html_validator(content):
#     return

# def filter(content):
#     if "dns resolution error" in content.lower():
#         return None 
    
#     return content

def fetch_and_embed_css(url, navigation_timeout=2000, request_timeout=10):
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
        
urls = []
for i in range(2):
    c4 = "../c4/c4-validation.0000{}-of-00008.json".format(str(i))
    with open(c4, 'r') as f:
        for line in f:
            d = json.loads(line)
            urls.append(d["url"])

urls = list(set(urls))
print ("total #urls: ", len(urls))

counter = 1
for i, url in tqdm(enumerate(urls)):
    html_content = fetch_and_embed_css(url)
    if html_content:
        with open("../c4-val-html/{}.html".format(counter), "w", encoding="utf-8") as f:
            f.write(html_content)
        counter += 1
    