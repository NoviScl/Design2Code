python3 claude.py  --prompt_method direct_prompting --model claude-3-5-sonnet-20240620 --file_name all  --subset testset_final  --take_screenshot

python3 claude.py  --prompt_method text_augmented_prompting --model claude-3-5-sonnet-20240620 --file_name all  --subset testset_final  --take_screenshot

python3 claude.py  --prompt_method revision_prompting --model claude-3-5-sonnet-20240620 --file_name all  --subset testset_final --orig_output_dir "claude-3-5-sonnet-20240620_text_augmented_prompting" --take_screenshot