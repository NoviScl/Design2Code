import base64
from PIL import Image
import os

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

     