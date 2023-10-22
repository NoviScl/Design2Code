from playwright.sync_api import sync_playwright
import requests
import base64

def fetch_and_embed_css(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Navigate to the URL
        page.goto(url)

        # Extract stylesheets' hrefs using JavaScript
        stylesheets_hrefs = page.eval_on_selector_all("link[rel='stylesheet']", 'nodes => nodes.map(n => n.href)')

        inline_css = ""
        # Fetch and embed each external CSS
        content = page.content()
        for href in stylesheets_hrefs:
            css_content = requests.get(href).text
            # print (href)
            # print (css_content)
            inline_css += '\n' + css_content + '\n\n'

        if "<style>" in content:
            content = content.replace("<style>", "<style>\n" + inline_css)
        else:
            content = content.replace('<head>', '<head>\n<style>' + inline_css + "</style>")

        browser.close()

        return content

# Example usage
html_content = fetch_and_embed_css("https://stevenyzzhang.github.io/website/")
with open("yanzhe.html", "w", encoding="utf-8") as f:
    f.write(html_content)