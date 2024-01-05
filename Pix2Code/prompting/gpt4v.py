import requests
import os
from tqdm import tqdm
from Pix2Code.data_utils.screenshot import take_screenshot
from gpt4v_utils import cleanup_response, rescale_image_loader

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
	response = response["choices"][0]["message"]["content"].strip()
	
	response = cleanup_response(response)

	return response

if __name__ == "__main__":
	# OpenAI API Key
	with open("../../api_key.txt") as f:
		api_key = f.read().strip()
	
	## load the prompt 
	with open("gpt_4v_prompt.txt") as f:
		prompt = f.read().strip()

	test_data_dir = "../../testset_100"
	predictions_dir = "../../predictions_100/gpt4v"
	for filename in tqdm(os.listdir(test_data_dir)):
		if filename.endswith("2.png"):
			## call GPT-4V
			# try:
			html = gpt4v_call(api_key, os.path.join(test_data_dir, filename), prompt)
			with open(os.path.join(predictions_dir, filename.replace(".png", ".html")), "w") as f:
				f.write(html)
			take_screenshot(os.path.join(predictions_dir, filename.replace(".png", ".html")), os.path.join(predictions_dir, filename))
			# except:
			# 	continue 
