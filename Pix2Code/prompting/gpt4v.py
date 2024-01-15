import requests
import os
from tqdm import tqdm
from Pix2Code.data_utils.screenshot import take_screenshot
from gpt4v_utils import cleanup_response, rescale_image_loader, gpt_cost
import json

def gpt4v_call(api_key, image_path, prompt):
	# Getting the base64 string
	base64_image = rescale_image_loader(image_path)

	headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {api_key}"
	}

	payload = {
		"model": "gpt-4-vision-preview",
		"messages": [
		{
			"role": "user",
			"content": [
			{
				"type": "text",
				"text": prompt
			},
			{
				"type": "image_url",
				"image_url": {
				"url": f"data:image/jpeg;base64,{base64_image}",
				"detail": "high"
				}
			}
			]
		}
		],
		"max_tokens": 3200
	}

	response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
	response = response.json()
	prompt_tokens, completion_tokens, cost = gpt_cost("gpt-4-vision-preview", response)
	response = response["choices"][0]["message"]["content"].strip()
	response = cleanup_response(response)

	return response, prompt_tokens, completion_tokens, cost

if __name__ == "__main__":
	## track usage
	## open "usage.json" if already exists; otherwise, create a new one
	if os.path.exists("usage.json"):
		with open("usage.json", 'r') as f:
			usage = json.load(f)
		total_prompt_tokens = usage["total_prompt_tokens"]
		total_completion_tokens = usage["total_completion_tokens"]
		total_cost = usage["total_cost"]
	else:
		total_prompt_tokens = 0 
		total_completion_tokens = 0 
		total_cost = 0

	# OpenAI API Key
	with open("../../api_key.txt") as f:
		api_key = f.read().strip()
	
	## load the prompt 
	with open("gpt_4v_prompt.txt") as f:
		prompt = f.read().strip()

	test_data_dir = "../../testset_100"
	predictions_dir = "../../predictions_100/gpt4v"
	for filename in tqdm(os.listdir(test_data_dir)):
		if filename == "2.png":
			## call GPT-4V
			# try:
			html, prompt_tokens, completion_tokens, cost = gpt4v_call(api_key, os.path.join(test_data_dir, filename), prompt)
			total_prompt_tokens += prompt_tokens
			total_completion_tokens += completion_tokens
			total_cost += cost
			# with open(os.path.join(predictions_dir, filename.replace(".png", ".html")), "w") as f:
			# 	f.write(html)
			# take_screenshot(os.path.join(predictions_dir, filename.replace(".png", ".html")), os.path.join(predictions_dir, filename))
			# # except:
			# # 	continue 
	
	## save usage
	usage = {
		"total_prompt_tokens": total_prompt_tokens,
		"total_completion_tokens": total_completion_tokens,
		"total_cost": total_cost
	}
	with open("usage.json", 'w') as f:
		usage = json.dump(usage, f, indent=4)
