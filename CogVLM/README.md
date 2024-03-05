# Design2Code Finetuning

We keep a snapshot of the CogAgent repo here, which we use for finetuning and inference.

The code base is based on CogVLM/CogAgent and swissarmytransformer v0.4.10.

Please install necessary libraries following the instruction in [CogVLM/CogAgent README](CogAgent_README.md).

The finetuning script is [finetune_cogagent_lora_design2code.sh](finetune_demo/finetune_cogagent_lora_design2code.sh).

Note that the LoRA modules are only added to the language decoder.

We provide [the example inference script](finetune_demo/inference_design2code.py) and the [Design2Code-18B-v0 model weight](https://huggingface.co/SALT-NLP/Design2Code).