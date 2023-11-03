from playwright.sync_api import sync_playwright
import os
from PIL import Image

def turn_image_red(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to RGB if it's not
        img = img.convert("RGB")
        
        # Load the pixel data
        pixels = img.load()

        # Iterate over each pixel and set it to red
        for i in range(img.width):
            for j in range(img.height):
                # Set the pixel to red
                pixels[i, j] = (255, 0, 0)

        # Save the image with the original name
        img.save(image_path)


def turn_image_blue(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to RGB if it's not
        img = img.convert("RGB")
        
        # Load the pixel data
        pixels = img.load()

        # Iterate over each pixel and set it to red
        for i in range(img.width):
            for j in range(img.height):
                # Set the pixel to red
                pixels[i, j] = (0, 0, 255)

        # Save the image with the original name
        img.save(image_path)


def take_screenshot(url, output_file="screenshot.png"):
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    with sync_playwright() as p:
        # Choose a browser, e.g., Chromium, Firefox, or WebKit
        browser = p.chromium.launch()
        page = browser.new_page()

        # Navigate to the URL
        page.goto(url)

        # Take the screenshot
        page.screenshot(path=output_file, full_page=True)

        browser.close()


image_path = './trial_dataset/rick.jpg'
with Image.open(image_path) as img:
    # Convert the image to RGB if it's not
    img = img.convert("RGB")

    take_screenshot("./trial_dataset/diyi.html", "./syn_dataset/diyi.png")
    take_screenshot("./trial_dataset/diyi_gpt4.html", "./syn_dataset/diyi_gpt4.png")

    turn_image_blue(image_path)
    take_screenshot("./trial_dataset/diyi.html", "./syn_dataset/diyi_blue.png")
    take_screenshot("./trial_dataset/diyi_gpt4.html", "./syn_dataset/diyi_gpt4_blue.png")

    turn_image_red(image_path)
    take_screenshot("./trial_dataset/diyi.html", "./syn_dataset/diyi_red.png")
    take_screenshot("./trial_dataset/diyi_gpt4.html", "./syn_dataset/diyi_gpt4_red.png")

    img.save(image_path)