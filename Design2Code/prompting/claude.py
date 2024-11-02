import sys, os
sys.path.append("/nlp/scr/zyanzhe/Pix2Code")

from tqdm import tqdm
from Design2Code.data_utils.screenshot import take_screenshot
from gpt4v_utils import cleanup_response, encode_image, gpt_cost, extract_text_from_html, index_text_from_html
import json
import anthropic
import argparse
import retry
import shutil
import mimetypes
import base64
from PIL import Image
import PIL

def encode_image(image_path: str, max_size: int = 8000):
    """Encodes an image to base64, rescaling if necessary, and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    # Open the image and resize if needed
    with Image.open(image_path) as img:
        width, height = img.size
        if width > max_size or height > max_size:
            # Calculate the new size maintaining the aspect ratio
            if width > height:
                new_height = int((max_size / width) * height)
                new_size = (max_size, new_height)
            else:
                new_width = int((max_size / height) * width)
                new_size = (new_width, max_size)

            img = img.resize(new_size, PIL.Image.LANCZOS)
            img.save("temp.png")

            with open("temp.png", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return {"type": "base64", "media_type": mime_type, "data": encoded_string}
            os.remove("temp.png")
        else:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return {"type": "base64", "media_type": mime_type, "data": encoded_string}

    return {"type": "base64", "media_type": mime_type, "data": encoded_string}


@retry.retry(tries=2, delay=2)
def claude_call(client, model, encoded_image, prompt):
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": encoded_image,
                    }
                ],
            }
        ],
    )
    response = cleanup_response(message.content[0].text)

    return response

@retry.retry(tries=2, delay=2)
def claude_revision_call(client, model, encoded_image_ref, encoded_image_pred, prompt):
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.,
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
                        "type": "image",
                        "source": encoded_image_ref,
                    },
                    {
                        "type": "text",
                        "text": "Current Webpage:"
                    },
                    {
                        "type": "image",
                        "source": encoded_image_pred,
                    }
                ],
            }
        ],
    )
    response = cleanup_response(message.content[0].text)

    return response

def direct_prompting(client, model, image_file):
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
    image = encode_image(image_file)

    ## call GPT-4V
    html = claude_call(client, model, image, direct_prompt)

    return html

def text_augmented_prompting(client, model, image_file):
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
    image = encode_image(image_file)

    ## call GPT-4V
    html = claude_call(client, model, image, text_augmented_prompt)

    return html

def visual_revision_prompting(client, model, input_image_file, original_output_image):
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

    html = claude_revision_call(client, model, input_image, original_output_image, prompt)

    return html

def layout_marker_prompting(client, model, image_file, auto_insertion=False):
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
    html = claude_call(client, model, orig_input_image, text_augmented_prompt)

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
        html = claude_call(client, model, orig_input_image, text_augmented_prompt)

    # ## remove the marker files
    # os.remove(image_file.replace(".png", "_marker.html"))
    # os.remove(image_file.replace(".png", "_marker.png"))

    return html

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_method', type=str, default='text_augmented_prompting', help='prompting method to be chosen from {direct_prompting, text_augmented_prompting, revision_prompting, layout_marker_prompting}')
    parser.add_argument('--orig_output_dir', type=str, default='gpt4v_text_augmented_prompting', help='directory of the original output that will be further revised')
    parser.add_argument('--model', type=str, default='claude-3-opus-20240229', help='model to benchmark')
    parser.add_argument('--file_name', type=str, default='all', help='any particular file to be tested')
    parser.add_argument('--subset', type=str, default='testset_100', help='evaluate on the full testset or just a subset (choose from: {testset_100, testset_full})')
    parser.add_argument('--take_screenshot', action="store_true", help='whether to render and take screenshot of the webpages')
    parser.add_argument('--auto_insertion', type=bool, default=False, help='whether to automatically insert texts into marker positions')
    args = parser.parse_args()

    # load API Key
    with open("../../api_key.json", "r") as f:
        api_key = json.load(f)
    
    ## set up client
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=api_key["anthropic_api_key"],
    )
    
    test_data_dir = "/juice2/scr2/nlp/pix2code/zyanzhe/github_filtered_hard_v0.4_full"
    cache_dir = "/juice2/scr2/nlp/pix2code/zyanzhe/Pix2Code/hard_v0.4_full/"

    if args.prompt_method == "direct_prompting":
      predictions_dir = cache_dir + f"{args.model}_direct_prompting"
    elif args.prompt_method == "text_augmented_prompting":
      predictions_dir = cache_dir + f"{args.model}_text_augmented_prompting"
    elif args.prompt_method == "layout_marker_prompting":
      predictions_dir = cache_dir + f"{args.model}_layout_marker_prompting" + ("_auto_insertion" if args.auto_insertion else "") 
    elif args.prompt_method == "revision_prompting":
      predictions_dir = cache_dir + f"{args.model}_visual_revision_prompting"
      orig_data_dir = cache_dir + args.orig_output_dir
    else: 
      print ("Invalid prompt method!")
      exit()
    
    ## create cache directory if not exists
    os.makedirs(predictions_dir, exist_ok=True)
    shutil.copy("rick.jpg", os.path.join(predictions_dir, "rick.jpg"))

    # get the list of predictions already made
    existing_predictions = [item for item in os.listdir(predictions_dir)]
    print ("#existing predictions: ", len(existing_predictions))
    
    test_files = []
    if args.file_name == "all":
      test_files = [item for item in os.listdir(test_data_dir) if item.endswith(".png") and "_marker" not in item and item not in existing_predictions]
    else:
      test_files = [args.file_name]

    counter = 0
    for filename in tqdm(test_files):
        # print (filename)

        if os.path.exists(os.path.join(predictions_dir, filename.replace(".png", ".html"))) and os.path.exists(os.path.join(predictions_dir, filename)):
            print(filename, "already exist!")
            counter += 1
            continue
        
        try:
            if args.prompt_method == "direct_prompting":
                html = direct_prompting(client, args.model, os.path.join(test_data_dir, filename))
            elif args.prompt_method == "text_augmented_prompting":
                html = text_augmented_prompting(client, args.model, os.path.join(test_data_dir, filename))
            elif args.prompt_method == "revision_prompting":
                html = visual_revision_prompting(client, args.model, os.path.join(test_data_dir, filename), os.path.join(orig_data_dir, filename))
            elif args.prompt_method == "layout_marker_prompting":
                html = layout_marker_prompting(client, args.model, os.path.join(test_data_dir, filename), auto_insertion=args.auto_insertion)

            with open(os.path.join(predictions_dir, filename.replace(".png", ".html")), "w") as f:
                f.write(html)
            if args.take_screenshot:
                take_screenshot(os.path.join(predictions_dir, filename.replace(".png", ".html")), os.path.join(predictions_dir, filename))
            counter += 1
        except Exception as e:
            print(filename)
            print(f"Function failed after retries: {e}")
            continue
            # continue 

    print ("#new predictions: ", counter)
            