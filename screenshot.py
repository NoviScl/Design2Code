from playwright.sync_api import sync_playwright

def take_screenshot(url, output_file="screenshot.png"):
    with sync_playwright() as p:
        # Choose a browser, e.g., Chromium, Firefox, or WebKit
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Navigate to the URL
        page.goto(url)
        
        # Take the screenshot
        page.screenshot(path=output_file, full_page=True)
        
        browser.close()

# Example usage
take_screenshot("https://cs.stanford.edu/~diyiy/research.html", "diyi_screenshot.png")