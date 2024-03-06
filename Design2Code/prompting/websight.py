import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format
from gpt4v_utils import cleanup_response
from tqdm import tqdm 
import os

DEVICE = torch.device("cuda")
HF_TOKEN = "YOUR HF TOKEN"

PROCESSOR = AutoProcessor.from_pretrained(
    "HuggingFaceM4/VLM_WebSight_finetuned",
    token=HF_TOKEN,
    cache_dir="/juice2/scr2/nlp/pix2code/huggingface"
)
MODEL = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceM4/VLM_WebSight_finetuned",
    token=HF_TOKEN,
    cache_dir="/juice2/scr2/nlp/pix2code/huggingface",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(DEVICE)

print ("parameter count: ", MODEL.num_parameters())

image_seq_len = MODEL.config.perceiver_config.resampler_n_latents
BOS_TOKEN = PROCESSOR.tokenizer.bos_token
BAD_WORDS_IDS = PROCESSOR.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

# The processor is the same as the Idefics processor except for the BILINEAR interpolation,
# so this is a hack in order to redefine ONLY the transform method
def custom_transform(x):
    x = convert_to_rgb(x)
    x = to_numpy_array(x)
    x = resize(x, (960, 960), resample=PILImageResampling.BILINEAR)
    x = PROCESSOR.image_processor.rescale(x, scale=1 / 255)
    x = PROCESSOR.image_processor.normalize(
        x,
        mean=PROCESSOR.image_processor.image_mean,
        std=PROCESSOR.image_processor.image_std
    )
    x = to_channel_dimension_format(x, ChannelDimension.FIRST)
    x = torch.tensor(x)
    return x

inputs = PROCESSOR.tokenizer(
    f"{BOS_TOKEN}<fake_token_around_image>{'<image>' * image_seq_len}<fake_token_around_image>",
    return_tensors="pt",
    add_special_tokens=False,
)


test_data_dir = "../../testset_final"
predictions_dir = "../../websight_predictions_full"

for filename in tqdm(os.listdir(test_data_dir)):
    if filename.endswith(".png"):
        image_path = os.path.join(test_data_dir, filename)
        with Image.open(image_path) as image:
            inputs["pixel_values"] = PROCESSOR.image_processor([image], transform=custom_transform)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        generated_ids = MODEL.generate(**inputs, bad_words_ids=BAD_WORDS_IDS, max_length=4096)
        generated_text = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_text = cleanup_response(generated_text)

        with open(os.path.join(predictions_dir, filename.replace(".png", ".html")), "w", encoding='utf-8') as f:
            f.write(generated_text)

