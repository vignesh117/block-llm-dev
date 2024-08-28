## BlockLLM

This repository is the official implementation of BlockLLM. It contains 2 experiments:

1. Finetuning on GLUE tasks
2. Pretraining on C4 


### Instructions for running the code

1. Install the requirements using the following command:
```bash
pip install -r exp-requirements.txt
```

2. Finetuning on GLUE tasks:
```bash
python run_glue.py \
    --model_name_or_path roberta-base \
    --task_name mrpc \
    --lora_all_modules \
    --max_length 512 \
    --seed=1234 \
    --lora_r 4 \
    --galore_scale 4 \
    --per_device_train_batch_size 16 \
    --update_proj_gap 500 \
    --learning_rate 3e-5 \
    --num_train_epochs 30 \
    --output_dir results/blockllm/roberta_base/mrpc \
    --with_tracking \
    --lr_scheduler_type linear
```

3. Pretraining on C4:
```bash
HF_HOME=/workspace/.cache python -m torch.distributed.run \
    --standalone \
    --nproc_per_node 1 \
    torchrun_greedy.py \
    --model_config configs/llama_60m.json \
    --lr 1e-3 \
    --rank 256 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --eval_every 500 \
    --optimizer blockllm \
    --warmup_steps 0
```

This code is based on the code released by https://github.com/jiaweizzhao/GaLore/

*Changes made to the original code:*
Added BlockLLM implementation, which involves modification to the following files
1. utils.py
2. run_glue.py
3. torchrun_main.py