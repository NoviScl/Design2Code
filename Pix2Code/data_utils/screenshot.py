from playwright.sync_api import sync_playwright
import os
from tqdm import tqdm

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

# for name in ["aryaman", "danqi", "diyi", "tatsu", "yanzhe"]:
#     take_screenshot("/Users/clsi/Desktop/Pix2Code/trial_dataset/" + "{}.html".format(name), "trial_dataset/" + "{}.png".format(name))

if __name__ == "__main__":
    predictions_dir = "../../predictions_100/finetuned_v0"
    for filename in tqdm(os.listdir(predictions_dir)):
        if filename.endswith(".html"):
            take_screenshot(os.path.join(predictions_dir, filename), os.path.join(predictions_dir, filename.replace(".html", ".png")))
