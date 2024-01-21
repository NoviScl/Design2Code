import requests
import os
from tqdm import tqdm
from Pix2Code.data_utils.screenshot import take_screenshot
from gpt4v_utils import cleanup_response, encode_image, gpt_cost, extract_text_from_html
import json
from openai import OpenAI, AzureOpenAI
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
		max_tokens=4000,
		temperature=0.0
	)

	prompt_tokens, completion_tokens, cost = gpt_cost("gpt-4-vision-preview", response.usage)
	response = response.choices[0].message.content.strip()
	response = cleanup_response(response)

	return response, prompt_tokens, completion_tokens, cost

def gpt4v_revision_call(openai_client, base64_image_ref, base64_image_pred, prompt):
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
						"type": "text", 
						"text": "Reference Webpage:"
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{base64_image_ref}",
							"detail": "high"
						},
					},
					{
						"type": "text", 
						"text": "Current Webpage:"
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{base64_image_pred}",
							"detail": "high"
						},
					},
				],
			}
		],
		max_tokens=4000,
		temperature=0.0
	)
	
	prompt_tokens, completion_tokens, cost = gpt_cost("gpt-4-vision-preview", response.usage)
	response = response.choices[0].message.content.strip()
	response = cleanup_response(response)

	return response, prompt_tokens, completion_tokens, cost

def gpt4_call(openai_client, prompt, model="gpt-4-1106", temperature=0., max_tokens=4000, json_output=False):
	prompt_messages = [{"role": "user", "content": prompt}]
	response_format = {"type": "json_object"} if json_output else {"type": "text"}
	completion = openai_client.chat.completions.create(
        model=model,
        messages=prompt_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format
    )
	cost = gpt_cost(model, completion.usage)
	response = completion.choices[0].message.content.strip()
	response = cleanup_response(response)
    
	return response, completion.usage.prompt_tokens, completion.usage.completion_tokens, cost

def direct_prompting(openai_client, image_file):
	'''
	{original input image + prompt} -> {output html}
	'''

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
	'''
	{original input image + extracted text + prompt} -> {output html}
	'''

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
	text_augmented_prompt += "Respond with the content of the HTML+CSS file (directly start with the code, do not add any additional explanation):\n"

	## encode image 
	base64_image = encode_image(image_file)

	## call GPT-4V
	html, prompt_tokens, completion_tokens, cost = gpt4v_call(openai_client, base64_image, text_augmented_prompt)

	return html, prompt_tokens, completion_tokens, cost

def text_revision_prompting(openai_client, input_html, original_html):
	'''
	TEXT ONLY
	{initial output html + oracle extracted text} -> {revised output html}
	'''
	extracted_texts = extract_text_from_html(input_html)

	prompt = ""
	prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
	prompt += "I have an HTML file for implementing a webpage but it is missing some elements. The current HTML implementation is:\n" + original_html + "\n\n"
	prompt += "I provide you all the texts that I want to include in the webpage here:\n"
	prompt += "\n".join(extracted_texts) + "\n\n"
	prompt += "Please revise and extend the given HTML file to include all the texts (unless there are parts that can't fit into the webpage appropriately) in the correct places or edit existing parts if they differ from the texts I provided. Make sure the code is syntactically correct and can render into a well-formed webpage. (\"rick.jpg\" is the placeholder image file.) "
	prompt += "Do not change the layout or style, just edit the content itself.\n"
	prompt += "Respond with the content of the new revised and improved HTML file:\n"

	response, prompt_tokens, completion_tokens, cost = gpt4_call(openai_client, prompt)

	return response, prompt_tokens, completion_tokens, cost

def visual_revision_prompting(openai_client, input_image_file, original_output_image):
	'''
	{input image + initial output image + initial output html + oracle extracted text} -> {revised output html}
	'''

	## load the original output
	with open(original_output_image.replace(".png", ".html"), "r") as f:
		original_output_html = f.read()

	## encode the image 
	input_image = encode_image(input_image_file)
	original_output_image = encode_image(original_output_image)

	## extract all texts from the webpage 
	with open(input_image_file.replace(".png", ".html"), "r") as f:
		html_content = f.read()
	texts = "\n".join(extract_text_from_html(html_content))

	prompt = ""
	prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
	prompt += "I have an HTML file for implementing a webpage but it is missing some elements:\n" + original_output_html + "\n\n"
	prompt += "I will provide the reference webpage that I want to build as well as the rendered webpage of the current implementation.\n"
	prompt += "I also provide you all the texts that I want to include in the webpage here:\n"
	prompt += "\n".join(texts) + "\n\n"
	prompt += "Please compare the two webpages and refer to the provided texts in to included, and revise the original HTML file to make it look exactly like the reference webpage. Make sure the code is syntactically correct and can render into a well-formed webpage. You can use \"rick.jpg\" as the placeholder image file.\n"
	prompt += "Respond directly with the content of the new revised and improved HTML file without any extra explanations:\n"

	html, prompt_tokens, completion_tokens, cost = gpt4v_revision_call(openai_client, input_image, original_output_image, prompt)

	return html, prompt_tokens, completion_tokens, cost

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--prompt_method', type=str, default='text_augmented_prompting', help='prompting method to be chosen from {direct_prompting, text_augmented_prompting}')
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

	# test_data_dir = "../../testset_100"
	# if args.prompt_method == "direct_prompting":
	# 	predictions_dir = "../../predictions_100/gpt4v_direct_prompting"
	# elif args.prompt_method == "text_augmented_prompting":
	# 	predictions_dir = "../../predictions_100/gpt4v_text_augmented_prompting"
	
	# ## visual revision 
	test_data_dir = "../../testset_100"
	orig_data_dir = "../../predictions_100/gpt4v_text_augmented_prompting"
	predictions_dir = "../../predictions_100/gpt4v_visual_revision_prompting"
	# for filename in tqdm(os.listdir(orig_data_dir)):
	# 	if filename == "102.html":
	# 	# if filename.endswith(".html"):
	# 		with open(os.path.join(test_data_dir, filename), "r") as f:
	# 			input_html_content = f.read()
	# 		with open(os.path.join(orig_data_dir, filename), "r") as f:
	# 			original_html_content = f.read()
	# 		# try:
	# 		# html, prompt_tokens, completion_tokens, cost = text_revision_prompting(openai_client, input_html_content, original_html_content)
	# 		html, prompt_tokens, completion_tokens, cost = visual_revision_prompting(openai_client, os.path.join(test_data_dir, filename.replace(".html", ".png")), os.path.join(orig_data_dir, filename.replace(".html", ".png")))
	# 		total_prompt_tokens += prompt_tokens
	# 		total_completion_tokens += completion_tokens
	# 		total_cost += cost

	# 		with open(os.path.join(predictions_dir, filename), "w") as f:
	# 			f.write(html)
	# 		take_screenshot(os.path.join(predictions_dir, filename), os.path.join(predictions_dir, filename.replace(".html", ".png")))
	# 		# except: 
	# 		# 	continue

	
	# with open("../../predictions_100/gpt4v_direct_prompting/2.html", "r") as f:
	# 	html_content = f.read()
	# response, cost = text_revision_prompting(personal_openai_client, html_content)
	# print (response, cost)

	for filename in tqdm(os.listdir(test_data_dir)):
		if filename == "00.png":
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
