import requests
import os
from tqdm import tqdm
from Pix2Code.data_utils.screenshot import take_screenshot
from gpt4v_utils import cleanup_response, encode_image, gpt_cost, extract_text_from_html
import json
from openai import AzureOpenAI
import argparse

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
		max_tokens=3200,
		temperature=0.0
	)

	# print (response)
	# print (response.choices[0].message.content.strip())
	# print (response.usage)
	prompt_tokens, completion_tokens, cost = gpt_cost("gpt-4-vision-preview", response.usage)
	response = response.choices[0].message.content.strip()
	response = cleanup_response(response)

	return response, prompt_tokens, completion_tokens, cost

def direct_prompting(openai_client, image_file):
	## the prompt 
	direct_prompt = ""
	direct_prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
	direct_prompt += "A user will provide you with a screenshot of a webpage.\n"
	direct_prompt += "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
	direct_prompt += "Include all CSS code in the HTML file itself.\n"
	direct_prompt += "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
	direct_prompt += "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
	direct_prompt += "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scrips for dynamic interactions.\n"
	direct_prompt += "Respond with the content of the HTML+CSS file:\n"
	
	## encode image 
	base64_image = encode_image(image_file)

	## call GPT-4V
	html, prompt_tokens, completion_tokens, cost = gpt4v_call(openai_client, base64_image, direct_prompt)

	return html, prompt_tokens, completion_tokens, cost

def text_augmented_prompting(openai_client, image_file):
	## encode the image 
	base64_image = encode_image(image_file)

	## extract all texts from the webpage 
	with open(image_file.replace(".png", ".html"), "r") as f:
		html_content = f.read()
	texts = "\n".join(extract_text_from_html(html_content))

	## the prompt
	text_augmented_prompt = ""
	text_augmented_prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
	text_augmented_prompt += "A user will provide you with a screenshot of a webpage, along with all texts that they want to put on the webpage.\n"
	text_augmented_prompt += "The text elements are:\n" + texts + "\n"
	text_augmented_prompt += "You should generate the correct layout structure for the webpage, and put the texts in the correct places. Not all text elements need to be used, just those that appear on the given screenshot.\n"
	text_augmented_prompt += "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
	text_augmented_prompt += "Include all CSS code in the HTML file itself.\n"
	text_augmented_prompt += "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
	text_augmented_prompt += "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
	text_augmented_prompt += "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scrips for dynamic interactions.\n"
	text_augmented_prompt += "Respond with the content of the HTML+CSS file:\n"

	## encode image 
	base64_image = encode_image(image_file)

	## call GPT-4V
	html, prompt_tokens, completion_tokens, cost = gpt4v_call(openai_client, base64_image, text_augmented_prompt)

	return html, prompt_tokens, completion_tokens, cost

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--prompt_method', type=str, default='direct_prompting', help='prompting method to be chosen from {direct_prompting, text_augmented_prompting}')
	args = parser.parse_args()

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

	test_data_dir = "../../testset_100"
	if args.prompt_method == "direct_prompting":
		predictions_dir = "../../predictions_100/gpt4v_direct_prompting"
	elif args.prompt_method == "text_augmented_prompting":
		predictions_dir = "../../predictions_100/gpt4v_text_augmented_prompting"
	
	for filename in tqdm(os.listdir(test_data_dir)):
		if filename.endswith("5.png"):
			try:
				if args.prompt_method == "direct_prompting":
					html, prompt_tokens, completion_tokens, cost = direct_prompting(openai_client, os.path.join(test_data_dir, filename))
				elif args.prompt_method == "text_augmented_prompting":
					html, prompt_tokens, completion_tokens, cost = text_augmented_prompting(openai_client, os.path.join(test_data_dir, filename))
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
