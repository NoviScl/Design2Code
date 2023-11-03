from tqdm import tqdm
from transformers import GPT2TokenizerFast
from bs4 import BeautifulSoup
import os
import re
import random 
random.seed(2023)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def remove_tags(html_content, tag="script"):
    while '<' + tag in html_content:
        start_index = html_content.find('<' + tag)
        end_index = html_content.find("</{}>".format(tag)) + len("</{}>".format(tag))  

        # Ensure both opening and closing tags are found
        if start_index == -1 or end_index < len("</{}>".format(tag))   or start_index >= end_index:
            break

        # Remove content between and including the script tags
        html_content = html_content[:start_index] + html_content[end_index:]

    return html_content 

def remove_html_comments(html_content):
    while "<!--" in html_content:
        start_index = html_content.find("<!--")
        end_index = html_content.find("-->") + 3  # +3 to account for the length of "-->"

        # Ensure both opening and closing comment delimiters are found
        if start_index == -1 or end_index < 3 or start_index >= end_index:
            break

        # Remove content between and including the comment delimiters
        html_content = html_content[:start_index] + html_content[end_index:]

    return html_content

def remove_css_js_comments(html_content):
    while "/*" in html_content:
        start_index = html_content.find("/*")
        end_index = html_content.find("*/") + 2  # +2 to account for the length of "*/"

        # Ensure both opening and closing comment delimiters are found
        if start_index == -1 or end_index < 2 or start_index >= end_index:
            break

        # Remove content between and including the comment delimiters
        html_content = html_content[:start_index] + html_content[end_index:]

    return html_content

def remove_extra_linebreaks(html_content):
    # Replace three or more consecutive line breaks with just two
    cleaned_string = re.sub(r'\n{3,}', '\n\n', html_content)
    return cleaned_string

def length_filter(html_content):
    ## filter too short pages
    html_len = len(tokenizer(html_content)["input_ids"])
    if html_len <= 100 or html_len >= 80000:
        return None
    
    return html_content

def html_validator(html_content):
    skip_words = ["404", "wordpress", "you have been blocked", "buy this domain", "403 "]
    for w in skip_words:
        if w.lower() in html_content.lower():
            return None

    return html_content

def remove_href_links(html_content):
    """
    Remove href attributes from <a> elements in the HTML that point to a web address.
    """

    href_pattern = 'href="'

    while href_pattern in html_content:
        start_index = html_content.find(href_pattern)
        if start_index == -1:
            break

        end_index = html_content.find('"', start_index + 6)  # +6 to skip past 'href="'

        if end_index != -1:
            html_content = html_content[:start_index] + html_content[end_index + 1:]
        else:
            break

    return html_content

def remove_link_tags(html_content):
    start_tag_pattern = '<link '
    href_pattern = 'href="'
    end_tag = '>'
    
    while start_tag_pattern in html_content:
        start_index = html_content.find(start_tag_pattern)
        href_index = html_content.find(href_pattern, start_index)
        end_index = html_content.find(end_tag, start_index)

        # If href is found and is before the end of the tag, then remove
        if start_index != -1 and href_index != -1 and href_index < end_index:
            html_content = html_content[:start_index] + html_content[end_index + len(end_tag):]
        else:
            break

    return html_content

def all_filters(html_content):
    html_content = html_validator(html_content)
    if not html_content:
        return None
    html_content = remove_extra_linebreaks(html_content)
    if len(html_content.split("\n")) <= 30 or len(html_content.split("\n")) >= 50000:
        return None
    html_content = remove_html_comments(html_content)
    html_content = remove_css_js_comments(html_content)
    html_content = remove_tags(html_content, tag="script")
    html_content = remove_tags(html_content, tag="audio")
    html_content = remove_tags(html_content, tag="video")
    html_content = remove_tags(html_content, tag="iframe")
    html_content = remove_tags(html_content, tag="map")
    html_content = remove_tags(html_content, tag="svg")
    html_content = remove_link_tags(html_content)
    html_content = remove_href_links(html_content)
    html_content = remove_extra_linebreaks(html_content)
    html_content = length_filter(html_content)
    if not html_content:
        return None
    
    return html_content

counter = 0
# for file in tqdm(os.listdir("c4-val-html")):
for idx in tqdm(range(1, 5136)):
    full_path = os.path.join("c4-val-html", str(idx)+".html")
    if os.path.isfile(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            html_content = f.read() 
            html_content = all_filters(html_content)
            if html_content:
                counter += 1
                with open("c4-val-html-cleaned/{}.html".format(idx), "w+", encoding="utf-8") as f:
                    f.write(html_content)
            
# print (counter)

# with open("c4-val-html/4970-cleaned.html", "w", encoding="utf-8") as f:
#     f.write(html_content)


