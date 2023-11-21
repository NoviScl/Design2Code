import base64
import requests
import json
import os
from tqdm import tqdm

prompt = '''You are an expert web developer who specializes in HTML and CSS.
A user will provide you with a screenshot of a webpage. 
You will return a single html file that uses HTML and CSS to create a fully functional static website.
Include any extra CSS in the HTML file itself.
If it involves any images, use \"rick.jpg\" as the placeholder.
Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.
Do not hallucinate any dependencies to external files. You do not need to include JavaScript. scrips for dynamic interactions.
Respond ONLY with the contents of the html file.'''

# Function to encode the image
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')

# # Path to your image
# image_path = "pilot_testset/11.png"

def gpt4v_call(api_key, image_path):
	# Getting the base64 string
	base64_image = encode_image(image_path)

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
	if response[ : 3] == "```":
		response = response[3 :]
	if response[-3 : ] == "```":
		response = response[ : -3]
	if response[ : 4] == "html":
		response = response[4 : ]

	return response

if __name__ == "__main__":
	# OpenAI API Key
	with open("api_key.txt") as f:
		api_key = f.read().strip()

	for filename in tqdm(os.listdir("pilot_testset")):
		if filename.endswith(".png"):
			## call GPT-4-V
			try:
				html = gpt4v_call(api_key, "pilot_testset/{}".format(filename))
				with open("pilot_testset/gpt4v_{}".format(filename.replace(".png", ".html")), "w") as f:
					f.write(html)
			except:
				continue 
