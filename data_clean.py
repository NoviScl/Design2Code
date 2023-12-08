from tqdm import tqdm
import nltk 
nltk.data.path.append("/juice2/scr2/nlp/pix2code/nltk_data")
from transformers import GPT2TokenizerFast
from nltk.tokenize import sent_tokenize
import os
import logging
import re
import json
import cssutils
from PIL import Image, ImageFile
from bs4 import BeautifulSoup,Tag, NavigableString
from screenshot import take_screenshot
from image_rescale import rescale_filter
import random 
random.seed(2023)


cssutils.log.setLevel(logging.CRITICAL)

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

def remove_useless_meta_tags(html_content):
    # A regular expression to match all meta tags
    meta_tags = re.findall(r'<meta[^>]+>', html_content, re.I)

    for tag in meta_tags:
        # Check for useful meta tag attributes
        if 'charset=' in tag.lower():
            continue  # Keep the charset meta tag
        if 'http-equiv=' in tag.lower():
            continue  # Keep the http-equiv meta tag
        if 'name="viewport"' in tag.lower():
            continue  # Keep the viewport meta tag

        # If none of the useful attributes are found, remove the tag
        html_content = html_content.replace(tag, '')

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
    split_lines = html_content.split("\n")
    split_lines = [line.replace("\n", "") for line in split_lines if (line is not None) and (line.replace(' ', '') != '')]
    cleaned_string = "\n".join(split_lines)
    return cleaned_string

def remove_web_links(html_content):
    pattern = (
        r'"'  # Opening quotation mark
        r'('
        r'(https?:\/\/|www\.)'  # "http:" or "https:" or "www."
        r'[^"\s]+'  # Any character except space and quotation mark
        r')'
        r'"'  # Closing quotation mark
    )
    cleaned_text = re.sub(pattern, '""', html_content)
    return cleaned_text

def length_filter(html_content, max_token=32000):
    ## filter too short pages
    html_len = len(tokenizer(html_content)["input_ids"])
    if html_len <= 100 or html_len >= max_token:
        return None
    
    return html_content, html_len

def html_validator(html_content):
    skip_words = ["404", "wordpress", "you have been blocked", "buy this domain", "403 ", "page not found", "squarespace "]
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

def remove_srcset_links(html_content):
    """
    Remove srcset links to external sources
    """

    href_pattern = 'srcset="'

    while href_pattern in html_content:
        start_index = html_content.find(href_pattern)
        if start_index == -1:
            break

        end_index = html_content.find('"', start_index + len('srcset="'))  

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

def remove_object_dependency(html_content):
    # List of picture file extensions
    picture_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg'}

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all object tags
    object_tags = soup.find_all('object')

    for obj in object_tags:
        data_attr = obj.get('data', '').lower()
        # If the data attribute is an image, replace it with 'rick.jpg'
        if any(data_attr.endswith(ext) for ext in picture_extensions):
            obj['data'] = 'rick.jpg'
        else:
            # If the data is not a picture, remove the object element
            obj.decompose()

    # Return the modified HTML content as a string
    return soup.prettify(formatter="html5")

def remove_embed_dependency(html_content):
    # List of picture file extensions
    picture_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg'}

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all object tags
    object_tags = soup.find_all('embed')

    for obj in object_tags:
        data_attr = obj.get('src', '').lower()
        # If the data attribute is an image, replace it with 'rick.jpg'
        if any(data_attr.endswith(ext) for ext in picture_extensions):
            obj['data'] = 'rick.jpg'
        else:
            # If the data is not a picture, remove the object element
            obj.decompose()

    # Return the modified HTML content as a string
    return soup.prettify(formatter="html5")

def item_truncation(html_content, max_items=4):
    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find and truncate lists
    lists = soup.find_all(['ul', 'ol'])
    for lst in lists:
        list_items = lst.find_all('li', recursive=False)  # only direct children
        if len(list_items) > max_items:
            for item in list_items[max_items:]:
                item.decompose()  # Remove excess list items

    # Find and truncate tables
    tables = soup.find_all('table')
    for table in tables:
        # print (table)
        # print ("----------------")
        rows = table.find_all('tr')
        # print (rows)
        if len(rows) > max_items:
            for row in rows[max_items:]:
                row.decompose()  # Remove excess rows
    
    # Return the modified HTML as a string
    return soup.prettify(formatter="html5")

def text_truncation(html_content, max_sent=2):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all elements
    all_elements = soup.find_all()

    # Process each element to find leaf nodes
    for element in all_elements:
        if not any(isinstance(child, Tag) for child in element.children):
            # It's a leaf node, now check if it's a visible tag
            if element.name in ['p', 'span', 'div', 'a', 'li', 'code', 'dd', 'em', 'td', 'th', 'u', 'strong', 'del', 'ins', 'b', 'i']:
                text = ''.join(element.stripped_strings)  # Get the text content of the element
                # print (element.name)
                # print (text)
                # print ("----------------")
                sentences = sent_tokenize(text)
                # If the block is longer than max_sent sentences, truncate it
                if len(sentences) > max_sent:
                    truncated_text = ' '.join(sentences[:max_sent])
                    for content in element.contents:
                        if not isinstance(content, Tag):
                            content.replace_with('')
                    element.append(truncated_text)
    html_content = soup.prettify(formatter="html5")

    # Convert the soup object to a list of lines
    lines = str(html_content).split('\n')

    # Process each line
    for i in range(1, len(lines) - 1):  # Skip the first and last line
        prev_line, current_line, next_line = lines[i - 1], lines[i], lines[i + 1]

        # Check if the current line is a text line between two tags
        if ('<' not in current_line) and ('>' not in current_line) and ('>' in prev_line) and ('<' in next_line):
            # print (current_line)
            # Current line is a text line
            text = current_line.strip()
            sentences = sent_tokenize(text)
            if len(sentences) > max_sent:
                truncated_text = ' '.join(sentences[ : (max_sent //2)])
                lines[i] = truncated_text

    # Reassemble the modified lines into HTML
    modified_html = '\n'.join(lines)
    soup = BeautifulSoup(modified_html, 'html.parser')
    
    return soup.prettify(formatter="html5")

def remove_unused_css(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all style tags
    style_tags = soup.find_all('style')

    for style_tag in style_tags:
        # Parse the CSS
        try:
            css = cssutils.parseString(style_tag.string)
            new_css_rules = []

            for rule in css:
                if isinstance(rule, cssutils.css.CSSStyleRule):
                    selector_is_used = False
                    for selector in rule.selectorList:
                        try:
                            # If the CSS selector matches any elements in the HTML, it is used
                            if soup.select(selector.selectorText):
                                selector_is_used = True
                                break
                        except:
                            # An error occurred while trying to match the selector
                            # print(f"A selector caused an error and was ignored: {e}")
                            # Consider selector as used to avoid removing potentially valid CSS
                            selector_is_used = True
                            break
                    
                    if selector_is_used:
                        # Keep the rule since the selector is used
                        new_css_rules.append(rule.cssText)
                else:
                    # Keep non-style rules (like @media, @keyframes, etc.) as they might be in use
                    new_css_rules.append(rule.cssText)

            # Replace the old CSS with only the used rules
            if new_css_rules:
                style_tag.string = '\n'.join(new_css_rules)
            else:
                # If no CSS rules are used, remove the style tag altogether
                style_tag.decompose()

        except:
            # print(f"An error occurred while parsing CSS: {e}")
            # In case of an error, leave the original CSS unchanged
            continue

    # Return the modified HTML content
    return soup.prettify(formatter="html5")

def remove_import(html_content):
    # Remove lines that start with "@import"
    new_lines = [line for line in html_content.split("\n") if not line.strip().startswith("@import")]
    return "\n".join(new_lines)

def all_filters_train(html_content):
    html_content = html_validator(html_content)
    if not html_content:
        return None
    # html_content = remove_extra_linebreaks(html_content)
    if len(html_content.split("\n")) <= 40 or len(html_content.split("\n")) >= 10000:
        return None
    html_content = remove_html_comments(html_content)
    html_content = remove_css_js_comments(html_content)
    html_content = remove_unused_css(html_content)
    html_content = remove_useless_meta_tags(html_content)
    html_content = remove_tags(html_content, tag="script")
    html_content = remove_tags(html_content, tag="audio")
    html_content = remove_tags(html_content, tag="video")
    html_content = remove_tags(html_content, tag="iframe")
    html_content = remove_tags(html_content, tag="map")
    html_content = remove_tags(html_content, tag="svg")
    html_content = remove_object_dependency(html_content)
    html_content = remove_embed_dependency(html_content)
    html_content = remove_link_tags(html_content)
    html_content = remove_href_links(html_content)
    html_content = remove_srcset_links(html_content)
    html_content = text_truncation(html_content)
    html_content = remove_extra_linebreaks(html_content)
    html_content = length_filter(html_content)
    if not html_content:
        return None
    
    return html_content

def all_filters_test(html_content):
    global total_len
    html_content = html_validator(html_content)
    if not html_content:
        return None
    if len(html_content.split("\n")) <= 40 or len(html_content.split("\n")) >= 10000:
        return None
    
    try:
        html_content = remove_tags(html_content, tag="script")
        html_content = remove_tags(html_content, tag="audio")
        html_content = remove_tags(html_content, tag="video")
        html_content = remove_tags(html_content, tag="iframe")
        html_content = remove_tags(html_content, tag="map")
        html_content = remove_tags(html_content, tag="svg")
        html_content = remove_object_dependency(html_content)
        html_content = remove_embed_dependency(html_content)
        html_content = remove_link_tags(html_content)
        html_content = remove_href_links(html_content)
        html_content = remove_srcset_links(html_content)
        html_content = item_truncation(html_content)
        html_content = text_truncation(html_content)
        # print (len(html_content.split("\n")))
        html_content = remove_import(html_content)
        html_content = remove_web_links(html_content)
        html_content, html_len = length_filter(html_content, max_token=320000)
        # print (len(html_content.split("\n")))
        # print ("----------------")
        # print (html_content)
        
        if not html_content:
            return None
    except:
        return None
    
    total_len += html_len
    return html_content


if __name__ == "__main__":
    global total_len 
    total_len = 0
    counter = 0
    all_url_dict = {}

    # for idx in range(8):
    #     print ("now processing: ", "/nlp/scr/clsi/c4-val-html-part{}".format(idx))
    #     with open("/nlp/scr/clsi/Pix2Code/url_dict_part{}.json".format(idx), "r") as f:
    #         url_dict = json.load(f)

    ## filtered set from round 1
    with open("/juice2/scr2/nlp/pix2code/auto_filtered.json", "r") as f:
        filtered_idx = json.load(f)
    with open("/juice2/scr2/nlp/pix2code/auto_filtered_part2.json", "r") as f:
        filtered_idx.extend(json.load(f))
    
    # print ("total number of webpages: ", len(filtered_idx))

    for file in tqdm(filtered_idx):
        if file.endswith(".html") and file in filtered_idx:
            full_path = os.path.join("/juice2/scr2/nlp/pix2code/testset_filter_round1", file)
            if os.path.isfile(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    # url = url_dict[file]
                    html_content = f.read() 
                    # print (html_content)
                    html_content = all_filters_test(html_content)
                    if html_content:
                        with open("/juice2/scr2/nlp/pix2code/testset_cleaned/{}".format(file), "w+", encoding="utf-8") as f:
                            f.write(html_content)
                        
                        ## take screenshot and rescale 
                        rescaled = rescale_filter("/juice2/scr2/nlp/pix2code/testset_cleaned", file)
                        if not rescaled:
                            ## if rescaled image doesn't pass filter, delete 
                            # print ("filtered out because of size or color")
                            os.remove("/juice2/scr2/nlp/pix2code/testset_cleaned/{}".format(file))
                            os.remove("/juice2/scr2/nlp/pix2code/testset_cleaned/{}".format(file.replace(".html", ".png")))
                        else:
                            counter += 1

                        # all_url_dict["{}.html".format(counter)] = url
                    # else:
                    #     print ("filtered out by length")
                    
    print ("total number of webpages: ", counter)
    # print ("avg number of tokens: ", total_len / counter)

    # with open("/juice2/scr2/nlp/pix2code/testset_filter_round1_url_dict.json", "w+") as f:
    #     json.dump(all_url_dict, f, indent=4)

