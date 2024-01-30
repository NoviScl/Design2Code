from Pix2Code.metrics.visual_score import visual_eval_v1, visual_eval_v3
from tqdm import tqdm 
import os
import sys
sys.stdout = open('eval_results.txt', 'w')

# predictions= ["finetuned_v0", "websight", "gpt4v_direct_prompting", "gpt4v_text_augmented_prompting", "gpt4v_visual_revision_prompting"]
predictions= ["gpt4v_layout_marker_prompting", "gpt4v_layout_marker_prompting_auto_insertion"]


reference_dir = "../../testset_100"

def print_multi_score(multi_score):
    final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score = multi_score
    print()
    print("final_size_score", final_size_score)
    print("Matched Text Score", final_matched_text_score)
    print("Position Score", final_position_score)
    print("Text Color Score", final_text_color_score)
    print("CLIP Score", final_clip_score)
    print("--------------------------------\n")

# for filename in ["16635.html", "8512.html", "13775.html"]:
#     print(filename)

#     ## websight score 
#     matched, final_score, multi_score = visual_eval_v3(os.path.join(websight_predictions_dir, filename.replace(".html", ".png")), os.path.join(reference_dir, filename.replace(".html", ".png")))
#     print ("websight score: ", final_score)
#     print_multi_score(multi_score)
    
for pred in predictions:
    all_scores = 0
    size_scores = 0
    text_scores = 0
    position_scores = 0
    color_scores = 0
    clip_scores = 0

    counter = 0
    predictions_dir = os.path.join("../../predictions_100", pred)
    print (predictions_dir)
    for filename in tqdm([item for item in os.listdir(predictions_dir) if item.endswith(".html")]):
        if filename.endswith(".html"):
            # try:
            #     # matched, loc_score, size_score, final_score = visual_eval(os.path.join(predictions_dir, filename.replace(".html", ".png")), os.path.join(reference_dir, filename.replace(".html", ".png")))
            #     matched, final_score = visual_eval_v1(os.path.join(predictions_dir, filename.replace(".html", ".png")), os.path.join(reference_dir, filename.replace(".html", ".png")))
            #     # print (filename, matched, final_score)
            # except:
            #     final_score = 0

            matched, final_score, multi_score = visual_eval_v3(os.path.join(predictions_dir, filename.replace(".html", ".png")), os.path.join(reference_dir, filename.replace(".html", ".png")))
            all_scores += final_score
            final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score = multi_score

            size_scores += final_size_score
            text_scores += final_matched_text_score
            position_scores += final_position_score
            color_scores += final_text_color_score
            clip_scores += final_clip_score

            counter += 1

    print ("\n")
    print ("avg final score: ", all_scores / counter)
    print ("avg size score: ", size_scores / counter)
    print ("avg text score: ", text_scores / counter)
    print ("avg position score: ", position_scores / counter)
    print ("avg color score: ", color_scores / counter)
    print ("avg clip score: ", clip_scores / counter)
    print ("--------------------------------\n")


    