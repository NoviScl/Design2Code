# Design2Code: How Far Are We From Automating Front-End Engineering?

Quick Links:
[[Dataset]](https://huggingface.co/datasets/SALT-NLP/Design2Code) 
[[Model Checkpoint]](https://huggingface.co/SALT-NLP/Design2Code-18B-v0) 
[[Project Page]](https://salt-nlp.github.io/Design2Code/)
[[Paper]]()

## Overview

This is the official repo for our Design2Code project, maintained by the SALT lab from Stanford NLP. In this repo, we provide: 

- The Design2Code benchmark dataset for the task of converting visual design (screenshot) into code implementation, which consists of 484 real-world webpages from C4 (examples shown below).

- Code for running all automatic evaluation. 

- Code for running multimodal prompting experiments on GPT-4V and Gemini Pro Vision. 

- Code for finetuning and running inference on our open-source Design2Code-18B model. 

</br>


![](example.png)



## Set Up

### With pip 

### Without pip 

```bash
python3 setup.py install --user
```


## Data and Predictions

### Testset 

You can download the full testset from this [Google Drive link](https://drive.google.com/file/d/1VdwCF5kuuYn4Otwy8WfzyHPwIjyC65cf/view?usp=sharing) or access it from the Huggingface dataset [page](https://huggingface.co/datasets/SALT-NLP/Design2Code).

After you unzip it into `testset_final/`, the folder should include 484 pairs of screenshots (`xx.png`) and corresponding HTML code (`xx.html`). We also include the placeholder image file `rick.jpg` which is used in the HTML codes.

### Taking Screenshots

In case you want to take screenshots of webpages by yourself, you can do so by running:

```bash
python3 data_utils/screenshot.py 
```

## Running Prompting Experiments 

## Running Inference on Design2Code-18B

## Finetuning Design2Code-18B

## Running Automatic Evaluation

## Other Functions

- `data_utils` contains the filtering and processing scripts to construct the test data from C4. 

- `metrics` contains the metric scripts for the evaluation.

- `prompting` contains the code for running all the prompting experiments, including the actual prompts used.

## License

The data, code and model checkpoint are intended and licensed for research use only. Please do not use them for any malicious purposes.

The benchmark is built on top of the C4 dataset, under the ODC Attribution License (ODC-By). 

## Acknowledgement

Our testset is filtered from [C4](https://huggingface.co/datasets/c4), training examples are sampled from [Websight](https://huggingface.co/datasets/HuggingFaceM4/WebSight). Our model is finetuned based on [CogAgent](https://github.com/THUDM/CogVLM). Thanks for their awsome work!
