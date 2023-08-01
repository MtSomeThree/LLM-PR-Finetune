# LLM-PR-Finetune
finetune large language model via controllable generation

Preliminary Experiments for GPT-2 related experiments

## Setup

Download RTP datasets. We will use ./rtp/prompts.jsonl

Apply for the access to PerspectiveAPI, and replace the API_KEY in perspectiveAPI.py

## Usage

Train NADO given base model, sampled data, and [optional] initialization
```
python train_nado.py --load_dir [base_model_dir] --load_dir2 [init_model_dir] --save_dir [save_NADO_dir] --samples_file [samples_dir]
```
check the code for other related configurations

Finetune GPT-2 given base model + NADO
```
python train_nado.py --finetune --load_dir [NADO_dir] --load_dir2 [base_model_dir] --save_dir [finetune_dir] --samples_file [samples_dir] 
```

Evaluate a model, model_type = {"gpt", "nado", "base"}
```
python train_nado.py --eval_model [model_type] --load_dir [model_dir]
```
