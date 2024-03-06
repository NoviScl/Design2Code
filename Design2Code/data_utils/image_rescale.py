from PIL import Image, ImageFile
import os
from tqdm import tqdm 
import json
from bs4 import BeautifulSoup,Tag, NavigableString
from nltk.tokenize import sent_tokenize
from screenshot import take_screenshot

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

def rescale_filter(dir, html_path):
    with open(os.path.join(dir, html_path), "r", encoding="utf-8") as f:
        html_content = f.read() 
    
    # html_content = item_truncation(html_content)
    # html_content = text_truncation(html_content)

    # ## save the truncated html
    # with open(os.path.join(dir, html_path), "w", encoding="utf-8") as f:
    #     f.write(html_content)

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
        if blue > 0.65:
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
            filtered = rescale_filter(directory, "{}.html".format(i))
            if filtered:
                selected.append("{}.html".format(i))
        except: 
            continue

    ## save selected indices
    with open(os.path.join("/juice2/scr2/nlp/pix2code/", "auto_filtered_part2.json"), "w") as f:
        json.dump(selected, f, indent=4)

