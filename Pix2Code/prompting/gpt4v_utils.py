import base64
from PIL import Image
import os
from bs4 import BeautifulSoup


def cleanup_response(response):
    ## simple post-processing
    if response[ : 3] == "```":
        response = response[3 :].strip()
    if response[-3 : ] == "```":
        response = response[ : -3].strip()
    if response[ : 4] == "html":
        response = response[4 : ].strip()
		
    ## strip anything after '</html>'
    if '</html>' in response:
        response = response.split('</html>')[0] + '</html>'
    return response 


# Function to encode the image
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


def rescale_image_loader(image_path):
    """
    Load an image, rescale it so that the short side is 768 pixels.
    If after rescaling, the long side is more than 2000 pixels, return None.
    If the original short side is already shorter than 768 pixels, no rescaling is done.

    Args:
    image_path (str): The path to the image file.

    Returns:
    Image or None: The rescaled image or None if the long side exceeds 2000 pixels after rescaling.
    """
    with Image.open(image_path) as img:
        # Get original dimensions
        width, height = img.size

        # Determine the short side
        short_side = min(width, height)
        long_side = max(width, height)

        # Check if resizing is needed
        if short_side <= 768:
            if long_side > 2000:
                print ("Bad aspect ratio for GPT-4V: ", image_path)
                return None
            else:
                ## no need rescaling, return the base64 encoded image
                return encode_image(image_path)

        # Calculate new dimensions
        scaling_factor = 768 / short_side
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        # Check if the long side exceeds 2000 pixels after rescaling
        if new_width > 2000 or new_height > 2000:
            print ("Bad aspect ratio for GPT-4V: ", image_path)
            return None

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        ## save to a temporary file
        resized_img = resized_img.save(image_path.replace(".png", "_rescaled.png"))
        base64_image = encode_image(image_path.replace(".png", "_rescaled.png"))
        os.remove(image_path.replace(".png", "_rescaled.png"))
        
        return base64_image


def gpt_cost(model, usage):
    '''
    Example response from GPT-4V: {'id': 'chatcmpl-8h0SZYavv8pmLGp45y05VB6NgzHxN', 'object': 'chat.completion', 'created': 1705260563, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 903, 'completion_tokens': 2, 'total_tokens': 905}, 'choices': [{'message': {'role': 'assistant', 'content': '```html'}, 'finish_reason': 'length', 'index': 0}]}
    '''
    if model == "gpt-4-vision-preview":
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cost = 0.01 * prompt_tokens / 1000 + 0.03 * completion_tokens / 1000
        return prompt_tokens, completion_tokens, cost 
    else:
        print ("model not supported: ", model)
        return 0


def remove_css_from_html(html_content):
    """
    Removes all CSS (contents within <style> and </style> tags) from an HTML
    webpage (provided as a string) and returns the modified HTML without CSS.

    :param html_content: A string containing the HTML content.
    :return: A string representing the HTML content without CSS.
    """
    # Using BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Finding and removing all <style> tags along with their contents
    [style_tag.decompose() for style_tag in soup.find_all('style')]

    return str(soup)


def extract_text_from_html(html_content):
    """
    Extracts all text elements from an HTML webpage (provided as a string)
    and returns a list of these text elements.

    :param html_content: A string containing the HTML content.
    :return: A list of strings, each representing a text element from the HTML content.
    """
    html_content_without_css = remove_css_from_html(html_content)
    soup = BeautifulSoup(html_content_without_css, 'html.parser')

    # Finding all text elements, excluding those within <script> tags
    texts = [element.strip() for element in soup.find_all(string=True) if element.parent.name != 'script' and len(element.strip()) > 0 and element.strip() != 'html']

    return texts

if __name__ == "__main__":
    test_data_dir = "../../testset_100"
    with open(os.path.join(test_data_dir, "58.html"), "r") as f:
        html_content = f.read()
    print (extract_text_from_html(html_content))