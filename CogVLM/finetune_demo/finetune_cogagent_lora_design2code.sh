#! /bin/bash
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

NUM_GPUS_PER_WORKER=$1
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="cogagent-chat"
VERSION="chat"
MODEL_ARGS="--from_pretrained $MODEL_TYPE \
    --max_length 4096 \
    --lora_rank 8 \
    --use_lora \
    --local_tokenizer lmsys/vicuna-7b-v1.5 \
    --version $VERSION"
# TIPS: max_length include low-resolution image sequence (which has 256 tokens) 

OPTIONS_SAT="SAT_HOME=/path/to/.sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER"
HOST_FILE_PATH="hostfile"

# "WebSight_164k_train_swap" is a subset of websight where we sampled 20% and swapped the position of HTML style and body. You can also replace this with your downloaded websight dataset.
# The current dataset is defined as HTMLDataset in /CogVLM/utils/utils/dataset.py
train_data='/path/to/WebSight_164k_train_swap'
valid_data='/path/to/WebSight_8k_val'

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 5000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --valid-data ${valid_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --vit_checkpoint_activations \
       --save-interval 5000 \
       --eval-interval 5000 \
       --save "/path/to/cogagent/checkpoints" \
       --eval-iters 10 \
       --eval-batch-size 1 \
       --split 1. \
       --deepspeed_config test_config_bf16.json \
       --skip-init \
       --seed 2023
"

              

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_cogagent_demo.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
