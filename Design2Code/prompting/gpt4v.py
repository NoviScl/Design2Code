import os
from tqdm import tqdm
from Design2Code.data_utils.screenshot import take_screenshot
from gpt4v_utils import cleanup_response, encode_image, gpt_cost, extract_text_from_html, index_text_from_html
import json
from openai import OpenAI, AzureOpenAI
import argparse
import retry
import shutil 

@retry.retry(tries=3, delay=2)
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
		max_tokens=4096,
		temperature=0.0,
		seed=2024
	)

	prompt_tokens, completion_tokens, cost = gpt_cost("gpt-4-vision-preview", response.usage)
	response = response.choices[0].message.content.strip()
	response = cleanup_response(response)

	return response, prompt_tokens, completion_tokens, cost

@retry.retry(tries=3, delay=2)
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
		max_tokens=4096,
		temperature=0.0,
		seed=2024
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
        response_format=response_format,
		seed=2024
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
	direct_prompt += "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n"
	direct_prompt += "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
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

	## extract all texts from the webpage 
	with open(image_file.replace(".png", ".html"), "r") as f:
		html_content = f.read()
	texts = "\n".join(extract_text_from_html(html_content))

	## the prompt
	text_augmented_prompt = ""
	text_augmented_prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
	text_augmented_prompt += "A user will provide you with a screenshot of a webpage, along with all texts that they want to put on the webpage.\n"
	text_augmented_prompt += "The text elements are:\n" + texts + "\n"
	text_augmented_prompt += "You should generate the correct layout structure for the webpage, and put the texts in the correct places so that the resultant webpage will look the same as the given one.\n"
	text_augmented_prompt += "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
	text_augmented_prompt += "Include all CSS code in the HTML file itself.\n"
	text_augmented_prompt += "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
	text_augmented_prompt += "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
	text_augmented_prompt += "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n"
	text_augmented_prompt += "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
	text_augmented_prompt += "Respond with the content of the HTML+CSS file (directly start with the code, do not add any additional explanation):\n"

	## encode image 
	base64_image = encode_image(image_file)

	## call GPT-4V
	html, prompt_tokens, completion_tokens, cost = gpt4v_call(openai_client, base64_image, text_augmented_prompt)

	return html, prompt_tokens, completion_tokens, cost

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
	prompt += "I have an HTML file for implementing a webpage but it has some missing or wrong elements that are different from the original webpage. The current implementation I have is:\n" + original_output_html + "\n\n"
	prompt += "I will provide the reference webpage that I want to build as well as the rendered webpage of the current implementation.\n"
	prompt += "I also provide you all the texts that I want to include in the webpage here:\n"
	prompt += "\n".join(texts) + "\n\n"
	prompt += "Please compare the two webpages and refer to the provided text elements to be included, and revise the original HTML implementation to make it look exactly like the reference webpage. Make sure the code is syntactically correct and can render into a well-formed webpage. You can use \"rick.jpg\" as the placeholder image file.\n"
	prompt += "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
	prompt += "Respond directly with the content of the new revised and improved HTML file without any extra explanations:\n"

	html, prompt_tokens, completion_tokens, cost = gpt4v_revision_call(openai_client, input_image, original_output_image, prompt)

	return html, prompt_tokens, completion_tokens, cost

def layout_marker_prompting(openai_client, image_file, auto_insertion=False):
	'''
	{marker image + extracted text + prompt} -> {output html}
	'''

	orig_input_image = encode_image(image_file)

	## extract all texts from the webpage 
	with open(image_file.replace(".png", ".html"), "r") as f:
		html_content = f.read()
	marker_html_content, text_dict = index_text_from_html(html_content)

	#save the marker html content
	with open(image_file.replace(".png", "_marker.html"), "w") as f:
		f.write(marker_html_content)
	take_screenshot(image_file.replace(".png", "_marker.html"), image_file.replace(".png", "_marker.png"))
	oracle_marker_image = encode_image(image_file.replace(".png", "_marker.png"))

	texts = ""
	for index, text in text_dict.items():
		texts += f"[{index}] {text}\n"

	## the layout generation prompt
	text_augmented_prompt = ""
	text_augmented_prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
	text_augmented_prompt += "A user will provide you with a screenshot of a webpage where all text elements should be index markers.\n"
	text_augmented_prompt += "The original text elements are:\n" + texts + "\n"
	text_augmented_prompt += "You should generate the correct layout structure for the webpage, and put the markers in the correct places. Markers should be wrapped in square backets like \"[1]\".\n"
	text_augmented_prompt += "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
	text_augmented_prompt += "Include all CSS code in the HTML file itself.\n"
	text_augmented_prompt += "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
	text_augmented_prompt += "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
	text_augmented_prompt += "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n"
	text_augmented_prompt += "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
	text_augmented_prompt += "Respond with the content of the HTML+CSS file (directly start with the code, do not add any additional explanation):\n"

	## call GPT-4V
	html, prompt_tokens, completion_tokens, cost = gpt4v_call(openai_client, orig_input_image, text_augmented_prompt)

	if auto_insertion:
		## put texts back into marker positions 
		for index, text in text_dict.items():
			html = html.replace(f"[{index}]", text)
	else:
		## take screenshot of the generated marker webpage
		with open(image_file.replace(".png", "_marker.html"), "w") as f:
			f.write(html)
		take_screenshot(image_file.replace(".png", "_marker.html"), image_file.replace(".png", "_marker.png"))
		generated_marker_image = encode_image(image_file.replace(".png", "_marker.png"))

		## the text insertion prompt
		text_augmented_prompt = ""
		text_augmented_prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
		text_augmented_prompt += "A user will provide you with a screenshot of a webpage. The implementation of this webpage with markers is:\n\n"
		text_augmented_prompt += html + "\n"
		text_augmented_prompt += "The original text elements are:\n" + texts + "\n"
		text_augmented_prompt += "Your task is to insert the corresponding text elements back into the marker positions (replace all the markers with actual text content) so that the resultant webpage will look the same as the given one..\n"
		text_augmented_prompt += "You need to return a single html file that uses HTML and CSS. Include all CSS code in the HTML file itself.\n"
		text_augmented_prompt += "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
		text_augmented_prompt += "Directly edit the given HTML implementation. Do not change the layout structure of the webpage, just insert the text elements into appropriate positions.\n"
		text_augmented_prompt += "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n"
		text_augmented_prompt += "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
		text_augmented_prompt += "Respond with the content of the HTML+CSS file (directly start with the code, do not add any additional explanation):\n"

		## call GPT-4V
		html, prompt_tokens, completion_tokens, cost = gpt4v_call(openai_client, orig_input_image, text_augmented_prompt)

	# ## remove the marker files
	# os.remove(image_file.replace(".png", "_marker.html"))
	# os.remove(image_file.replace(".png", "_marker.png"))

	return html, prompt_tokens, completion_tokens, cost

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--prompt_method', type=str, default='text_augmented_prompting', help='prompting method to be chosen from {direct_prompting, text_augmented_prompting, revision_prompting, layout_marker_prompting}')
	parser.add_argument('--orig_output_dir', type=str, default='gpt4v_text_augmented_prompting', help='directory of the original output that will be further revised')
	parser.add_argument('--file_name', type=str, default='all', help='any particular file to be tested')
	parser.add_argument('--subset', type=str, default='testset_100', help='evaluate on the full testset or just a subset (choose from: {testset_100, testset_full})')
	parser.add_argument('--take_screenshot', action="store_true", help='whether to render and take screenshot of the webpages')
	parser.add_argument('--auto_insertion', type=bool, default=False, help='whether to automatically insert texts into marker positions')
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
	with open("../api_key.json", "r") as f:
		api_key = json.load(f)
	
	openai_client = AzureOpenAI(
		api_key=api_key["openai_key"],
		api_version="2023-12-01-preview",
		azure_endpoint=api_key["openai_endpoint"]
	)

	## specify file directory 
	if args.subset == "testset_final":
		test_data_dir = "../testset_final"
		cache_dir = "../predictions_final/"
	elif args.subset == "testset_100":
		test_data_dir = "../testset_100"
		cache_dir = "../predictions_100/"
	elif args.subset == "testset_full":
		test_data_dir = "../testset_full"
		cache_dir = "../predictions_full/"
	else:
		print ("Invalid subset!")
		exit()

	if args.prompt_method == "direct_prompting":
		predictions_dir = cache_dir + "gpt4v_direct_prompting"
	elif args.prompt_method == "text_augmented_prompting":
		predictions_dir = cache_dir + "gpt4v_text_augmented_prompting"
	elif args.prompt_method == "layout_marker_prompting":
		predictions_dir = cache_dir + "gpt4v_layout_marker_prompting" + ("_auto_insertion" if args.auto_insertion else "") 
	elif args.prompt_method == "revision_prompting":
		predictions_dir = cache_dir + "gpt4v_visual_revision_prompting"
		orig_data_dir = cache_dir + args.orig_output_dir
	else: 
		print ("Invalid prompt method!")
		exit()
	
	## create cache directory if not exists
	os.makedirs(predictions_dir, exist_ok=True)
	shutil.copy(test_data_dir + "/rick.jpg", os.path.join(predictions_dir, "rick.jpg"))
	
	test_files = []
	if args.file_name == "all":
		test_files = [item for item in os.listdir(test_data_dir) if item.endswith(".png") and "_marker" not in item]
	else:
		test_files = [args.file_name]

	for filename in tqdm(test_files):
		if filename.endswith(".png"):
			print (filename)
			try:
				if args.prompt_method == "direct_prompting":
					html, prompt_tokens, completion_tokens, cost = direct_prompting(openai_client, os.path.join(test_data_dir, filename))
				elif args.prompt_method == "text_augmented_prompting":
					html, prompt_tokens, completion_tokens, cost = text_augmented_prompting(openai_client, os.path.join(test_data_dir, filename))
				elif args.prompt_method == "revision_prompting":
					html, prompt_tokens, completion_tokens, cost = visual_revision_prompting(openai_client, os.path.join(test_data_dir, filename), os.path.join(orig_data_dir, filename))
				elif args.prompt_method == "layout_marker_prompting":
					html, prompt_tokens, completion_tokens, cost = layout_marker_prompting(openai_client, os.path.join(test_data_dir, filename), auto_insertion=args.auto_insertion)

				total_prompt_tokens += prompt_tokens
				total_completion_tokens += completion_tokens
				total_cost += cost

				with open(os.path.join(predictions_dir, filename.replace(".png", ".html")), "w") as f:
					f.write(html)
				if args.take_screenshot:
					take_screenshot(os.path.join(predictions_dir, filename.replace(".png", ".html")), os.path.join(predictions_dir, filename), do_it_again=True)
			except:
				continue 

	## save usage
	usage = {
		"total_prompt_tokens": total_prompt_tokens,
		"total_completion_tokens": total_completion_tokens,
		"total_cost": total_cost
	}

	with open("usage.json", 'w+') as f:
		usage = json.dump(usage, f, indent=4)

