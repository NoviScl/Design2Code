import os
from playwright.sync_api import sync_playwright
import os

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

def process_files_with_prefix(folder, prefix, processing_function):
    """
    Process files in a folder that start with a given prefix using a specified function.

    Parameters:
    folder (str): The folder to search in.
    prefix (str): The prefix to filter files by.
    processing_function (function): The function to apply to each file.
    """
    # Check if the folder exists
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist.")
        return

    # List all files in the folder
    for filename in os.listdir(folder):
        # Check if the filename starts with the prefix
        if filename.startswith(prefix) and "demo" not in filename and "_p" not in filename and "png" not in filename:
            # Construct the full path
            file_path = os.path.join(folder, filename)
            processing_function(file_path, file_path.replace(".html", ".png"), True)

            file_path = file_path.replace("gpt4v_", "")

            # Process the file using the provided function
            processing_function(file_path, file_path.replace(".html", ".png"), True)

process_files_with_prefix("/Users/zhangyanzhe/Documents/GitHub/Pix2Code/pilot_testset", "gpt4v", take_screenshot)