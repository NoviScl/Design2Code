# Design2Code

__Design2Code: How Far Are We From Automating Front-End Engineering__

[[Dataset]](https://huggingface.co/datasets/SALT-NLP/Design2Code-hf) [[Model Weight]](https://huggingface.co/SALT-NLP/Design2Code-18B-v0) [[Project Page]](https://salt-nlp.github.io/Design2Code/)

*Chenglei Si, Yanzhe Zhang, Zhengyuan Yang, Ruibo Liu, Diyi Yang*

<details><summary>Abstract</summary>

Generative AI has made rapid advancements in recent years, achieving unprecedented capabilities in multimodal understanding and code generation. This enabled a brand new paradigm of front-end development, where multimodal LLMs can potentially convert visual designs into code implementations directly, thus automating the front-end engineering pipeline. In this work, we provide the first systematic study on this visual design to code implementation task (dubbed as Design2Code). We manually curate a benchmark of 484 real-world webpages as test cases and develop a set of automatic evaluation metrics to assess how well current multimodal LLMs can generate the code implementations that directly render into the given reference webpages, given the screenshots as input. We develop a suit of multimodal prompting methods and show their effectiveness on GPT-4V and Gemini Vision Pro. We also finetune an open-source Design2Code-18B model that successfully matches the performance of Gemini Vision Pro. Both human evaluation and automatic metrics show that GPT-4V is the clear winner on this task, where annotators think GPT-4V generated webpages can replace the original reference webpages in 49% cases in terms of visual appearance and content; and perhaps surprisingly, in 64% cases GPT-4V generated webpages are considered better than even the original reference webpages. Our fine-grained break-down metrics indicate that open-source models mostly lag in recalling visual elements from the input webpages and in generating correct layout designs, while aspects like text content and coloring can be drastically improved with proper finetuning.

</details>

## Overview

![](example.png)

## Installation


## Example Script


## Code Structure

- `data_utils` contains the filtering and processing scripts to construct the test data from C4. 

- `metrics` contains the metric scripts for the evaluation.

- `prompting` contains the code for running all the prompting experiments, including the actual prompts used.

## License

The data, code and model checkpoint are intended and licensed for research use only. Please do not use them for any malicious purposes.

The benchmark is built on top of the C4 dataset, under the ODC Attribution License (ODC-By). 

## Acknowledgement

Our testset is filtered from [C4](https://huggingface.co/datasets/c4), training examples are sampled from [Websight](https://huggingface.co/datasets/HuggingFaceM4/WebSight). Our model is finetuned based on [CogAgent](https://github.com/THUDM/CogVLM). Thanks for their awsome work!
