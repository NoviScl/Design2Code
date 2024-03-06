import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import argparse
import os 
from tqdm import tqdm

cache_dir = '/juice2/scr2/nlp/pix2code/huggingface'

direct_prompt = ""
# direct_prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
# direct_prompt += "A user will provide you with a screenshot of a webpage.\n"
# direct_prompt += "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
# direct_prompt += "Include all CSS code in the HTML file itself.\n"
# direct_prompt += "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
# direct_prompt += "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
# direct_prompt += "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scrips for dynamic interactions.\n"
# direct_prompt += "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
direct_prompt += "Write the HTML code."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()
    MODEL_PATH = args.from_pretrained
    TOKENIZER_PATH = args.local_tokenizer
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH, cache_dir=cache_dir)
    if args.bf16:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16

    print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

    if args.quant:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            trust_remote_code=True,
            cache_dir=cache_dir
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=args.quant is not None,
            trust_remote_code=True,
            cache_dir=cache_dir
        ).to(DEVICE).eval()

    print ("parameter count: ", model.num_parameters())

    test_dir = "/nlp/scr/clsi/Pix2Code/testset_final"
    prediction_dir = "/nlp/scr/clsi/Pix2Code/cogagent_predictions_full"
    
    for filename in tqdm(os.listdir(test_dir)):
        if filename.endswith(".png"):

            image_path = os.path.join(test_dir, filename)
            image = Image.open(image_path).convert('RGB')
            query = direct_prompt
            history = []

            print ("Doing inference on: ", image_path)
            input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])
            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
            }
            if 'cross_images' in input_by_model and input_by_model['cross_images']:
                inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

            # add any transformers params here.
            gen_kwargs = {"max_length": 2048,
                            # "temperature": 0.1,
                            "do_sample": False}

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(outputs[0])
                response = response.split("</s>")[0].strip()
            
            with open(os.path.join(prediction_dir, filename.replace(".png", ".html")), "w") as f:
                f.write(response)

