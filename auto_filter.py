from PIL import Image, ImageFile
import os
from tqdm import tqdm 
import json

def calculate_blue_percentage(image_path):
    with Image.open(image_path) as img:
        pixels = list(img.getdata())
        blue_count = sum(1 for pixel in pixels if (pixel[0] + pixel[1] < 5 and pixel[2] >= 250)) 

    blue_percentage = (blue_count / len(pixels)) * 100
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

passed = []
directory = "/juice2/scr2/nlp/pix2code/testset_filter_round1"
# for filename in tqdm(os.listdir(directory)):
for i in tqdm(range(1, 100)):
    filename = str(i) + ".png"
    # if filename.endswith(".png"):
    img_path = os.path.join(directory, filename)
    try:
        blue = calculate_blue_percentage(img_path)
        if blue >= 60:
            passed.append(filename)
    except: 
        continue

print (len(filename))
with open("auto_color_filtered.json", "w+") as f:
    json.dump(passed, d)

# blue_percentage = calculate_blue_percentage("/juice2/scr2/nlp/pix2code/testset_filter_round1/1.png")
# print(f"Blue pixels percentage: {blue_percentage}%")

