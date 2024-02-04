from tqdm import tqdm
import numpy as np
import json
import os

reference_dir = "../../testset_full"
test_dirs = {
    "gpt4v_direct_prompting": "../../gpt4v_predictions_full/gpt4v_direct_prompting",
    "gpt4v_text_augmented_prompting": "../../gpt4v_predictions_full/gpt4v_text_augmented_prompting",
    "gpt4v_visual_revision_prompting": "../../gpt4v_predictions_full/gpt4v_visual_revision_prompting",
    "gemini_direct_prompting": "../../gemini_predictions_full/gemini_direct_prompting",
    "gemini_text_augmented_prompting": "../../gemini_predictions_full/gemini_text_augmented_prompting",
    "gemini_visual_revision_prompting": "../../gemini_predictions_full/gemini_visual_revision_prompting"
}

valid_files = [filename for filename in os.listdir(reference_dir) if filename.endswith(".png")]
print ("total #egs: ", len(valid_files))

# ## check if the file is in all prediction directories
# for filename in os.listdir(reference_dir):
#     if filename.endswith(".html"):
#         if all([os.path.exists(os.path.join(test_dirs[key], filename.replace(".html", ".png"))) for key in test_dirs]):
#             file_name_list.append(filename)


