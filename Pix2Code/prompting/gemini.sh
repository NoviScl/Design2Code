# python3 gemini.py \
#  --prompt_method direct_prompting \
#  --file_name all \
#  --subset testset_full \
#  --take_screenshot

# python3 gemini.py \
#  --prompt_method text_augmented_prompting \
#  --file_name all \
#  --subset testset_full \
#  --take_screenshot

python3 gemini.py \
 --prompt_method revision_prompting \
 --file_name all \
 --subset testset_full \
 --orig_output_dir "gemini_text_augmented_prompting" \
 --take_screenshot
