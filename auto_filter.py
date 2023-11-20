from PIL import Image, ImageFile
import os
from tqdm import tqdm 
import json
from bs4 import BeautifulSoup,Tag, NavigableString
from nltk.tokenize import sent_tokenize
from screenshot import take_screenshot


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

def calculate_blue_percentage(img):
    # with Image.open(image_path) as img:
    pixels = list(img.getdata())
    blue_count = sum(1 for pixel in pixels if (pixel[0] + pixel[1] < 5 and pixel[2] >= 250)) 

    blue_percentage = (blue_count / len(pixels))
    return blue_percentage

def size_filter(image_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(image_path) as img:
        img.draft(None, (2005, 2005))
        img.load()
        short_side = min(img.width, img.height)
        long_side = max(img.width, img.height)
    if short_side < 768 and long_side < 2000:
        return True 
    return False 

def rescale_image(image_path):
    """
    Load an image, rescale it so that the short side is 768 pixels.
    If after rescaling, the long side is more than 2000 pixels, return None.
    If the original short side is already shorter than 768 pixels, no rescaling is done.

    Args:
    image_path (str): The path to the image file.

    Returns:
    Image or None: The rescaled image or None if the long side exceeds 2000 pixels after rescaling.
    """
    # try:
    # Open the image
    with Image.open(image_path) as img:
        # Get original dimensions
        width, height = img.size
        # print ("original: ", width, height)

        # Determine the short side
        short_side = min(width, height)
        long_side = max(width, height)

        # Check if resizing is needed
        if short_side <= 768:
            if long_side > 2000:
                return False
            else:
                # print ("retained: ", width, height)
                img = img.save(image_path)
                return True

        # Calculate new dimensions
        scaling_factor = 768 / short_side
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        # Check if the long side exceeds 2000 pixels after rescaling
        if new_width > 2000 or new_height > 2000:
            return False

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        # print ("resized: ", new_width, new_height)

        resized_img = resized_img.save(image_path)
        return True

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     return False

def auto_filter(dir, html_path):
    with open(os.path.join(dir, html_path), "r", encoding="utf-8") as f:
        html_content = f.read() 
    
    html_content = item_truncation(html_content)
    html_content = text_truncation(html_content)

    ## save the truncated html
    with open(os.path.join(dir, html_path), "w", encoding="utf-8") as f:
        f.write(html_content)

    ## take screenshot of the truncated webpage 
    take_screenshot(os.path.join(dir, html_path), os.path.join(dir, html_path.replace(".html", ".png")))
    # print (html_path, "screenshot saved")

    rescaled_img = rescale_image(os.path.join(dir, html_path.replace(".html", ".png")))
    if not rescaled_img:
        # print (html_path, rescaled_img)
        return False

    ## load the rescaled image 
    with Image.open(os.path.join(dir, html_path.replace(".html", ".png"))) as rescaled_img:
        # print (html_path, rescaled_img.size)
        ## blue is the placeholder image file; filter out cases where the entire webpage is just the image
        blue = calculate_blue_percentage(rescaled_img)
        # print ("blue: ", blue)
        if blue >= 0.7:
            # print (html_path, "filtered out by color")
            del rescaled_img
            return False

    # rescaled_img.save(os.path.join(dir, html_path.replace(".html", ".png")))
    # print (html_path, "updated")
    del rescaled_img

    return True


if __name__ == "__main__":
    directory = "/juice2/scr2/nlp/pix2code/testset_copy"
    selected = []
    for i in tqdm(range(5001, 17798)):
        try:
            filtered = auto_filter(directory, "{}.html".format(i))
            if filtered:
                selected.append("{}.html".format(i))
        except: 
            continue

    ## save selected indices
    with open(os.path.join("/juice2/scr2/nlp/pix2code/", "auto_filtered_part2.json"), "w") as f:
        json.dump(selected, f, indent=4)

