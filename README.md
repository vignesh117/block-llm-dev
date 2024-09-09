## BlockLLM

This repository is the official implementation of BlockLLM https://arxiv.org/abs/2406.17296. It contains 2 experiments:

1. Finetuning on GLUE tasks
2. Pretraining on C4 


### Instructions for running the code

1. Install the requirements using the following command:
```bash
pip install -r exp-requirements.txt
```

2. Install blockllm package:
```bash
pip install -e .
```

3. Finetuning on GLUE tasks:
```bash
python run_glue.py \
    --model_name_or_path roberta-base \
    --task_name mrpc \
    --max_length 512 \
    --seed=1234 \
    --per_device_train_batch_size 16 \
    --learning_rate 3e-5 \
    --num_train_epochs 30 \
    --output_dir results/blockllm/roberta_base/mrpc \
    --with_tracking \
    --lr_scheduler_type linear \
    --enable_blockllm \
    --sparsity_level 0.95 \
    --update_freq 500
```

3. Pretraining on C4:
```bash
HF_HOME=/workspace/.cache python -m torch.distributed.run \
    --standalone \
    --nproc_per_node 1 \
    torchrun_greedy.py \
    --model_config configs/llama_60m.json \
    --enable_blockllm \
    --sparsity_level 0.95 \
    --lr 1e-3 \
    --rank 256 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --eval_every 500 \
    --optimizer blockllm \
    --warmup_steps 0
```

## Integration with Llama-factory
We are aiming to get BlockLLM integrated into the Llama-factory repo. Our unofficial implementation can be found at https://github.com/vignesh117/LLaMA-Factory. Please see the examples in `examples/extras/blockllm`.

## Citation

Please cite our paper if you use this code in your work:
```
@article{zhao2024blockllm,
  title={BlockLLM: Efficient and Effective Sparse Training for Large Language Models},
  author={Zhao, Jiawei and Chen, Hao and Ma, Jiezhong and Liu, Yang and Wang, Yongwei and Wang, Hao and Wang, Hao and Wang, Hao},
  journal={arXiv preprint arXiv:2406.17296},
  year={2024}
}
```

This code is based on the code released by https://github.com/jiaweizzhao/GaLore/

*Changes made to the original code:*
Added BlockLLM implementation, which involves modification to the following files
1. utils.py
2. run_glue.py
3. torchrun_main.py