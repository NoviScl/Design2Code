# python3 gpt4v.py \
#  --prompt_method direct_prompting \
#  --file_name all \
#  --subset testset_full \
#  --take_screenshot


# python3 gpt4v.py \
#  --prompt_method text_augmented_prompting \
#  --file_name all \
#  --subset testset_full \
#  --take_screenshot


# python3 gpt4v.py \
#  --prompt_method revision_prompting \
#  --file_name all \
#  --subset testset_full \
#  --orig_output_dir "gpt4v_text_augmented_prompting" \
#  --take_screenshot

# python3 gpt4v.py \
#  --prompt_method layout_marker_prompting \
#  --file_name all \
#  --subset testset_100 \
#  --take_screenshot


python3 gpt4v.py \
 --prompt_method layout_marker_prompting \
 --file_name all \
 --subset testset_100 \
 --auto_insertion true \
 --take_screenshot

python3 eval.py

#  python3 gpt4v.py \
#  --prompt_method layout_marker_prompting \
#  --file_name all \
#  --subset testset_100 \
#  --auto_insertion \
#  --take_screenshot
