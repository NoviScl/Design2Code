import os
from tqdm import tqdm
from Design2Code.data_utils.screenshot import take_screenshot
from gpt4v_utils import cleanup_response, encode_image, gpt_cost, extract_text_from_html, index_text_from_html
import json
from openai import OpenAI, AzureOpenAI
import argparse
import retry
import shutil 

if __name__ == "__main__":
    predictions_dirs = ["../../gemini_pro_predictions_full/direct_prompting", "../../gemini_pro_predictions_full/text_augmented_prompting", "../../gemini_pro_predictions_full/visual_revision_prompting"]
    for predictions_dir in predictions_dirs:
        for filename in tqdm(os.listdir(predictions_dir)):
            if filename.endswith(".html"):
                with open(os.path.join(predictions_dir, filename), "r") as f:
                    html_content = f.read()
                cleaned_html = cleanup_response(html_content)
                with open(os.path.join(predictions_dir, filename), "w") as f:
                    f.write(cleaned_html)
                try:
                    take_screenshot(os.path.join(predictions_dir, filename), os.path.join(predictions_dir, filename.replace(".html", ".png")))
                except:
                    print ("screen shot failed for: ", filename)