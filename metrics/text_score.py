from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import re
import random 
random.seed(2023)

def extract_text(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')
    
    # Remove the title tag if it exists
    title_tag = soup.title
    if title_tag:
        title_tag.extract()

    all_text = soup.get_text()
    all_text = re.sub(r'\n{2,}', '\n', all_text)
    return all_text

def word_f1(true_paragraph, predicted_paragraph):
    # Tokenize paragraphs by space to get words
    true_words = set(true_paragraph.split())
    predicted_words = set(predicted_paragraph.split())
    
    # Compute number of common words
    common_words = true_words.intersection(predicted_words)
    
    # Compute precision and recall
    if not predicted_words:
        precision = 0.0
    else:
        precision = len(common_words) / len(predicted_words)
    
    if not true_words:
        recall = 0.0
    else:
        recall = len(common_words) / len(true_words)
    
    # Compute F1 score
    if precision + recall == 0:
        return 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        
    return f1

if __name__ == '__main__':
    with open("trial_dataset/yanzhe.html", "r", encoding="utf-8") as f:
        reference = f.read()
    with open("trial_dataset/yanzhe_gpt4.html", "r", encoding="utf-8") as f:
        prediction = f.read()

    print(word_f1(reference, prediction))


