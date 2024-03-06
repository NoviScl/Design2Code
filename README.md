# Design2Code: How Far Are We From Automating Front-End Engineering?

Quick Links:
[[Dataset]](https://huggingface.co/datasets/SALT-NLP/Design2Code-hf) 
[[Model Checkpoint]](https://huggingface.co/SALT-NLP/Design2Code-18B-v0) 
[[Project Page]](https://salt-nlp.github.io/Design2Code/)
[[Paper]](https://arxiv.org/abs/2403.03163)

## Overview

This is the official repo for our Design2Code project, maintained by the SALT lab from Stanford NLP. In this repo, we provide: 

- The Design2Code benchmark dataset for the task of converting visual design (screenshot) into code implementation, which consists of 484 real-world webpages from C4 (examples shown below).

- Code for running all automatic evaluation. 

- Code for running multimodal prompting experiments on GPT-4V and Gemini Pro Vision. 

- Code for finetuning and running inference on our open-source Design2Code-18B model. 


![](example.png)



## Set Up

All code is tested on Python 3.11. We recommend using a virtual environment to manage the dependencies.

Clone this repo and install the necessary libraries:

```bash
pip install -e .
```

Taking screenshots and running evaluations also need to install browsers

```bash
playwright install
```

If the above doesn't work, try:

```bash
python3 -m playwright install
```

## Data and Predictions

### Testset 

You can download the full testset from this [Google Drive link](https://drive.google.com/file/d/12uRO5EC7hkg6qAOyfJhb4YsrQ_qpL5bt/view?usp=sharing) or access it from the Huggingface dataset [page](https://huggingface.co/datasets/SALT-NLP/Design2Code).

After you unzip it into `testset_final/`, the folder should include 484 pairs of screenshots (`xx.png`) and corresponding HTML code (`xx.html`). We also include the placeholder image file `rick.jpg` which is used in the HTML codes.

### Taking Screenshots

In case you want to take screenshots of webpages by yourself, you can do so by running:

```bash
cd Design2Code
python3 data_utils/screenshot.py 
```

Remember to replace the file name or directory in the script with your own. 

### Model Predictions

To facilitate more analysis, we also release all model predictions on our benchmark:

- [GPT-4V](https://drive.google.com/file/d/1SgWL4E5uKVo-8D3Bj-VWvysJs_2OguA1/view?usp=sharing) (including Direct Prompting, Text-Augmented Prompting, and Self-Revision Prompting)
- [Gemini Pro Vision](https://drive.google.com/file/d/18cpGdL1Yhv9UU7odcqkncDItGo0Guuy_/view?usp=sharing) (including Direct Prompting, Text-Augmented Prompting, and Self-Revision Prompting)
- [WebSight VLM-8B](https://drive.google.com/file/d/1lFqLyJSDwZAEhZ4mhRqrK_-d5hhcrrEM/view?usp=sharing) (Huggingface)
- [Design2Code-18B](https://drive.google.com/file/d/1XxZMeVpAGu3fGvtKetHvk2bk3vyBcj2e/view?usp=sharing) (Ours)
- [Automatic Evaluation Results](https://drive.google.com/file/d/1qahQCmGqEXPXKmn2RzNwHsOI-CAQSP6P/view?usp=sharing)
- [Human Eval - Pairwise Model Comparison](https://huggingface.co/datasets/SALT-NLP/Design2Code_human_eval_pairwise)
- [Human Eval - Direct Assessment](https://huggingface.co/datasets/SALT-NLP/Design2Code_human_eval_reference_vs_gpt4v)

## Running Prompting Experiments 

To run prompting experiments, first put your OpenAI / Google Gemini API keys in a file called `api_keys.json` in the root directory. It should look like this:

```json
{
    "organization_id": "",
    "openai_key": "",
    "openai_endpoint": "",
    "gemini_api_key": ""
}
```

Then, to run GPT-4V experiments, run:

```bash
bash prompting/gpt4v.sh
```

To run Gemini Pro Vision experiments, run:

```bash
bash prompting/gemini.sh
```

The bash scripts include scripts for running Direct Prompting, Text-Augmented Prompting, and Self-Revision Prompting. All prompts are written in `prompting/gpt4v.py` and `prompting/gemini.py`, you can modify it to run your own prompts or develop smarter prompting strategies. We welcome any contributions to this part of the project! 

Also note that we are accessing the OpenAI API from Azure, and you might need some slight modification for directly calling the OpenAI API. 

### Running Inference on CogAgent-18B

We also provide code to run inference on the base model CogAgent-18B:

```bash
python3 prompting/cogagent.py
```

Be aware that the model is not finetuned on Design2Code, so the performance is very bad, often times not even producing valid HTML code.

## Running Inference on Design2Code-18B

The finetuned model is based on [CogAgent](./CogVLM/CogAgent_README.md), please install necessary libraries following the instructions.

You can run inference by:

```bash
python3 CogVLM/finetune_demo/inference_design2code.py
```

## Finetuning Design2Code-18B

The finetuning script is [finetune_cogagent_lora_design2code.sh](./CogVLM/finetune_demo/finetune_cogagent_lora_design2code.sh).

## Running Automatic Evaluation

You can use the following command to run automatic evaluation:

```bash
python3 metrics/multi_processing_eval.py
```

Note that you need to specify the directories where you store the model predictions in `metrics/multi_processing_eval.py` (starting at line 54), like the following:

```python
test_dirs = {
        "gpt4v_direct_prompting": "../predictions_final/gpt4v_direct_prompting",
        "gemini_direct_prompting": "../predictions_final/gemini_direct_prompting"
}
```

where we assume each directory in the dict contains the predictions of the corresponding model/method (i.e., each directory should contain 484 predicted HTML files for the full test set, or for some subset that you sampled for yourself). The script will compute scores for all automatic metrics for all examples in each directory and store the results in a dictionary, with the following format:

```python
{
    "gpt4v_direct_prompting": {
        "2.html": [0.1, 0.2, ...],
        "6.html": [0.3, 0.4, ...],
        ...
    },
    "gemini_direct_prompting": {
        "2.html": [0.5, 0.6, ...],
        "6.html": [0.7, 0.8, ...],
        ...
    }
}
```

where each list contains the fine-grained breakdown metrics. The script will also print the average scores for each model/method in the end, with the following format:

```
gpt4v_direct_prompting

Block-Match:  0.6240771561959276
Text:  0.9769471025300969
Position:  0.7787072741618328
Color:  0.7068853534416764
CLIP:  0.8924754858016968
--------------------------------

gemini_direct_prompting

Block-Match:  0.6697374012874602
Text:  0.9731735845969769
Position:  0.6502285758036523
Color:  0.8531304981602478
CLIP:  0.8571878373622894
--------------------------------
```

These metrics are also what we reported in the paper. By default, we support multiprocessing to speed up evaluation, you can also manually turn it off by setting `multiprocessing = False` on line 40.
For your reference, it can take up to 1 hour to run the the evaluation on the full testset (for each model/method). 


## Other Functions

- `data_utils` contains various filtering and processing scripts that we used to construct the test data from C4. 


## License

The data, code and model checkpoint are intended and licensed for research use only. Please do not use them for any malicious purposes.

The benchmark is built on top of the C4 dataset, under the ODC Attribution License (ODC-By). 


## Acknowledgement

Our testset is filtered from [C4](https://huggingface.co/datasets/c4), training examples are sampled from [Websight](https://huggingface.co/datasets/HuggingFaceM4/WebSight). Our model is finetuned based on [CogAgent](https://github.com/THUDM/CogVLM). Thanks for their awsome work!

If you find our work helpful, please consider citing our paper:

```
@misc{si2024design2code,
    title={Design2Code: How Far Are We From Automating Front-End Engineering?},
    author={Chenglei Si and Yanzhe Zhang and Zhengyuan Yang and Ruibo Liu and Diyi Yang},
    year={2024},
    eprint={2403.03163},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

We welcome all types of contributions to this project (PRs are welcome!). If you have any questions, please feel free to leave issues or email us.
