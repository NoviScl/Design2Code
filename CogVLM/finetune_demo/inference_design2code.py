import sys
sys.path.insert(1, '/path/to/CogVLM')
from sat.model import AutoModel
import argparse
from utils.models import CogAgentModel, CogVLMModel, FineTuneTestCogAgentModel
import torch
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel
from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import CogAgentModel, CogVLMModel
from tqdm import tqdm 
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--repetition_penalty', type=float, default=1.1)
args = parser.parse_args()
args.bf16 = True
args.stream_chat = False
args.version = "chat"

# You can download the testset from https://huggingface.co/datasets/SALT-NLP/Design2Code
test_data_dir = "/path/to/Design2Code"
predictions_dir = "/path/to/design2code_18b_v0_predictions"
if not os.path.exists(predictions_dir):
    try:
        os.makedirs(predictions_dir)
    except:
        pass

filename_list = [filename for filename in os.listdir(test_data_dir) if filename.endswith(".png")]
world_size = 1
model, model_args = FineTuneTestCogAgentModel.from_pretrained(
        f"/path/to/design2code-18b-v0",
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=0,
        rank=0,
        world_size=1,
        model_parallel_size=1,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True,
        device='cuda',
        bf16=True,
        fp16=None), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
model = model.eval()
model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else args.version
print("[Language processor version]:", language_processor_version)
tokenizer = llama2_tokenizer("lmsys/vicuna-7b-v1.5", signal_type=language_processor_version)
image_processor = get_image_processor(model_args.eva_args["image_size"][0])
cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
text_processor_infer = llama2_text_processor_inference(tokenizer, 2048, model.image_length)

def get_html(image_path):
    with torch.no_grad():
        history = None
        cache_image = None
        # We use an empty string as the query
        query = ''
    
        response, history, cache_image = chat(
            image_path,
            model,
            text_processor_infer,
            image_processor,
            query,
            history=history,
            cross_img_processor=cross_image_processor,
            image=cache_image,
            max_length=4096,
            top_p=1.0,
            temperature=args.temperature,
            top_k=1,
            invalid_slices=text_processor_infer.invalid_slices,
            repetition_penalty=args.repetition_penalty,
            args=args
        )
    
    return response

for filename in tqdm(filename_list):
    image_path = os.path.join(test_data_dir, filename)
    generated_text = get_html(image_path)
    with open(os.path.join(predictions_dir, filename.replace(".png", ".html")), "w", encoding='utf-8') as f:
        f.write(generated_text)
