from Pix2Code.metrics.visual_score import visual_eval_v1
from tqdm import tqdm 
import os

reference_dir = "../../testset_100"
predictions_dir = "../../predictions_100/gpt4v_text_augmented_prompting"
all_scores = 0
counter = 0

for filename in tqdm([item for item in os.listdir(predictions_dir) if item.endswith("2.html")]):
    if filename.endswith("2.html"):
        # matched, loc_score, size_score, final_score = visual_eval(os.path.join(predictions_dir, filename.replace(".html", ".png")), os.path.join(reference_dir, filename.replace(".html", ".png")))
        matched, final_score = visual_eval_v1(os.path.join(predictions_dir, filename.replace(".html", ".png")), os.path.join(reference_dir, filename.replace(".html", ".png")))
        print (filename, matched, final_score)
        all_scores += final_score
        counter += 1

print ("\n")
print ("avg score: ", all_scores / counter)
    