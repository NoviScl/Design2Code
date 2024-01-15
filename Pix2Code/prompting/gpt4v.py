import requests
import os
from tqdm import tqdm
from Pix2Code.data_utils.screenshot import take_screenshot
from gpt4v_utils import cleanup_response, encode_image, rescale_image_loader, gpt_cost
import json
from openai import AzureOpenAI

def gpt4v_call(openai_client, base64_image, prompt):
	response = openai_client.chat.completions.create(
		model="gpt-4-vision-preview",
		messages=[
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
						},
					},
				],
			}
		],
		max_tokens=3200
	)

	# print (response)
	# print (response.choices[0].message.content.strip())
	# print (response.usage)
	prompt_tokens, completion_tokens, cost = gpt_cost("gpt-4-vision-preview", response.usage)
	response = response.choices[0].message.content.strip()
	response = cleanup_response(response)

	return response, prompt_tokens, completion_tokens, cost

def direct_prompting(direct_prompt, openai_client, image_file):
	## encode image 
	base64_image = encode_image(image_file)

	## call GPT-4V
	html, prompt_tokens, completion_tokens, cost = gpt4v_call(openai_client, base64_image, direct_prompt)

	return html, prompt_tokens, completion_tokens, cost

def text_augmented_prompting(text_augmented_prompt, openai_client, image_file):
	## encode the image 
	base64_image = encode_image(image_file)

	## extract all texts from the webpage 

	return 

if __name__ == "__main__":
	## track usage
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

	## OpenAI API Key
	with open("../../api_key.json", "r") as f:
		api_key = json.load(f)
	
	openai_client = AzureOpenAI(
		api_key=api_key["salt_openai_key"],
		api_version="2023-12-01-preview",
		azure_endpoint=api_key["salt_openai_endpoint"]
	)

	## load the direct prompt 
	with open("gpt_4v_prompt.txt") as f:
		direct_prompt = f.read().strip()

	test_data_dir = "../../testset_100"
	predictions_dir = "../../predictions_100/gpt4v"
	for filename in tqdm(os.listdir(test_data_dir)):
		if filename == "6.png":
			try:
				html, prompt_tokens, completion_tokens, cost = direct_prompting(direct_prompt, openai_client, os.path.join(test_data_dir, filename))
				total_prompt_tokens += prompt_tokens
				total_completion_tokens += completion_tokens
				total_cost += cost

				with open(os.path.join(predictions_dir, filename.replace(".png", ".html")), "w") as f:
					f.write(html)
				take_screenshot(os.path.join(predictions_dir, filename.replace(".png", ".html")), os.path.join(predictions_dir, filename))
			except:
				continue 

	## save usage
	usage = {
		"total_prompt_tokens": total_prompt_tokens,
		"total_completion_tokens": total_completion_tokens,
		"total_cost": total_cost
	}
	with open("usage.json", 'w') as f:
		usage = json.dump(usage, f, indent=4)
