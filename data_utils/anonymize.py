import re
import random
import os
from screenshot import take_screenshot
from tqdm import tqdm 

def anonymize_html(html_content):
    # Regular expressions for different types of PII
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_regex = r'\(?\b\d{3}\)?[-.]?\s?\d{3}[-.]?\d{4}\b'
    # # Simple pattern for addresses (very basic)
    # address_regex = r'\b\d{1,6}\s+[A-Za-z0-9\s]{3,}\b'

    # Replace email addresses
    html_content = re.sub(email_regex, '[EMAIL REDACTED]', html_content)
    # Replace phone numbers
    html_content = re.sub(phone_regex, '[PHONE REDACTED]', html_content)
    # # Replace addresses
    # html_content = re.sub(address_regex, '[ADDRESS REDACTED]', html_content)

    return html_content

if __name__ == "__main__":
    dir = "testset_manual_filtered"
    target_dir = "testset_anonymized"

    # for filename in tqdm(os.listdir(dir)):
    #     if ".html" in filename: 
    #         with open(os.path.join(dir, filename), "r", encoding="utf-8") as f:
    #             html_content = f.read()
    #             html_content = anonymize_html(html_content)
            
    #         with open(os.path.join(target_dir, filename), "w+", encoding="utf-8") as f:
    #             f.write(html_content)
            
    #         take_screenshot(os.path.join(target_dir, filename), os.path.join(target_dir, filename.replace(".html", ".png")))

    ## re-take screenshots of manually anonymized testset
    for filename in tqdm(os.listdir(target_dir)):
        if ".html" in filename: 
            take_screenshot(os.path.join(target_dir, filename), os.path.join(target_dir, filename.replace(".html", ".png")))



