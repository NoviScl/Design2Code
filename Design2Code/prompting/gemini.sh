python3 prompting/gemini.py \
 --prompt_method direct_prompting \
 --file_name all \
 --subset testset_final \
 --take_screenshot

python3 prompting/gemini.py \
 --prompt_method text_augmented_prompting \
 --file_name all \
 --subset testset_final \
 --take_screenshot

python3 prompting/gemini.py \
 --prompt_method revision_prompting \
 --file_name all \
 --subset testset_final \
 --orig_output_dir "../predictions_final/gemini_text_augmented_prompting" \
 --take_screenshot
