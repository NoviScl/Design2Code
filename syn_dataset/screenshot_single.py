import os
from playwright.sync_api import sync_playwright
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description='Process two path strings.')

# Define the arguments
parser.add_argument('--html', type=str)
parser.add_argument('--png', type=str)

# Parse the arguments
args = parser.parse_args()

def take_screenshot(url, output_file="screenshot.png", do_it_again=False):
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    if os.path.exists(output_file) and not do_it_again:
        print(f"{output_file} exists!")
        return

    with sync_playwright() as p:
        # Choose a browser, e.g., Chromium, Firefox, or WebKit
        browser = p.chromium.launch()
        page = browser.new_page()

        # Navigate to the URL
        page.goto(url)

        # Take the screenshot
        page.screenshot(path=output_file, full_page=True)

        browser.close()

    print(f"Saved to {output_file}")

take_screenshot(args.html, args.png, do_it_again=True)