# SDLoRA — Experiments on Deep S4 Models

This directory contains experiments on deep S4 models using SDLoRA (State-Dimension LoRA).

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Run Experiments](#run-experiments)
   - [Synthetic Datasets (`run_s4.py`)](#run_s4py)
   - [Real-world Datasets (`run_s4_real.py`)](#run_s4_realpy)
3. [Example Usage](#example-usage)

## Environment Setup

Working directory: `ssm-research/S4`

1. Create the conda environment:
   ```bash
   conda env create -f conda_environment.yml
   ```

2. Create `environment.py` in this directory for wandb logging:
   ```python
   WANDB_INFO = {
       'entity': f'{YOUR_ENTITY}',
       'project': 's4-peft',
   }
   ```

## Run Experiments

There are two main scripts for running experiments:

- `run_s4.py`: For PEFT experiments on synthetic datasets
- `run_s4_real.py`: For PEFT experiments on real-world datasets

### `run_s4.py`

This script is used for synthetic dataset experiments. Key parameters include:

* `--n_epochs`: int, number of epochs
* `--batch_size`: int, batch size
* `--seq_len`: int, sequence length of the synthetic data
* `--lr`: learning rate
* `--wandb`: bool, whether to log to wandb
* `--warmup_epochs`: int, number of warmup epochs (only for SDLoRA & SDT)
* `--warmup_lr`: float, learning rate for the warmup phase (only for SDLoRA & SDT)
* `--overwrite`: bool, whether to overwrite the existing results
* `--d_model`: int, model dimension
* `--frozen_n_layers`: int, number of layers in the frozen model
* `--frozen_d_state`: int, dimension of the states in the frozen model
* `--dropout`: float, dropout rate
* `--task_type`: str, task type, either `regression` or `classification`
* `--lp_method`: str, parameter-efficient method applied to the linear layers, either `ours` or `lora` (`ours` is dimension selection method)
* `--D_method`: str, parameter-efficient method applied to the D (residual connection), either `full` or `freeze`
* `--ssm_method`: str, parameter-efficient method applied to the SSM, either `ours`, `full`, `freeze`, or `lora`
* `--our_mode`: str, mode of our method, either `hard` or `soft` (our paper uses `hard`, soft is for using L1-penalty for sparse training)
* `--select_states_dim`: int, dimension of the selected states (for `ours` method only, and only when `our_mode` is `hard`)
* `--updatable_states_dim`: int, dimension of the updatable states (for `ours` method only, and only when `our_mode` is `hard`)
* `--select_channels_dim`: int, dimension of the selected channels (for `ours` method only, and only when `our_mode` is `hard`)
* `--updatable_channels_dim`: int, dimension of the updatable channels (for `ours` method only, and only when `our_mode` is `hard`)
* `--select_states_penalty`: float, L1-penalty for the selected states (for `ours` method only, and only when `our_mode` is `soft`)
* `--updatable_states_penalty`: float, L1-penalty for the updatable states (for `ours` method only, and only when `our_mode` is `soft`)
* `--ssm_warmup_method`: str, warmup method for the SSM, either `full` or `ssm`
* `--data`: str, dataset, either `synthetic`, `mnist`, or `cifar10`
* `--target_n_layers`: int, number of layers in the target model (only for `synthetic` dataset)
* `--target_d_state`: int, dimension of the states in the target model (only for `synthetic` dataset)

#### Notes for SDLoRA Configuration:
- Set `ssm_method` to `ours`
- Set `our_mode` to `hard`
- Set `ssm_warmup_method` to `ssm`
- Set `D_method` to `full`
- Set `lp_method` to `lora`

### `run_s4_real.py`

This script is used for real-world dataset experiments. Key parameters include:

* `--n_epochs`: int, number of epochs
* `--batch_size`: int, batch size
* `--lr`: float, learning rate
* `--wandb`: bool, whether to log to wandb
* `--warmup_epochs`: int, number of warmup epochs (only for SDLoRA & SDT)
* `--warmup_lr`: float, learning rate for the warmup phase (only for SDLoRA & SDT)
* `--overwrite`: bool, whether to overwrite the existing results
* `--device`: str, device, either `cuda` or `cpu`
* `--pretrain_epochs`: int, number of pretrain epochs
* `--pretrain_lr`: float, learning rate for the pretrain phase
* `--d_model`: int, model dimension
* `--frozen_n_layers`: int, number of layers in the frozen model
* `--frozen_d_state`: int, dimension of the states in the frozen model
* `--dropout`: float, dropout rate
* `--task_type`: str, task type, either `regression` or `classification`
* `--lp_method`: str, parameter-efficient method applied to the linear layers, either `ours` or `lora` (`ours` is dimension selection method)
* `--D_method`: str, parameter-efficient method applied to the D (residual connection), either `full` or `freeze`
* `--ssm_method`: str, parameter-efficient method applied to the SSM, either `ours`, `full`, `freeze`, or `lora`
* `--ssm_lora_rank`: int, rank of the low-rank update for the SSM (only for `lora` method)
* `--our_mode`: str, mode of our method, either `hard` or `soft` (our paper uses `hard`, soft is for using L1-penalty for sparse training)
* `--select_states_dim`: int, dimension of the selected states (for `ours` method only, and only when `our_mode` is `hard`)
* `--updatable_states_dim`: int, dimension of the updatable states (for `ours` method only, and only when `our_mode` is `hard`)
* `--select_channels_dim`: int, dimension of the selected channels (for `ours` method only, and only when `our_mode` is `hard`)
* `--updatable_channels_dim`: int, dimension of the updatable channels (for `ours` method only, and only when `our_mode` is `hard`)
* `--select_states_penalty`: float, L1-penalty for the selected states (for `ours` method only, and only when `our_mode` is `soft`)
* `--updatable_states_penalty`: float, L1-penalty for the updatable states (for `ours` method only, and only when `our_mode` is `soft`)
* `--select_channels_penalty`: float, L1-penalty for the selected channels (for `ours` method only, and only when `our_mode` is `soft`)
* `--updatable_channels_penalty`: float, L1-penalty for the updatable channels (for `ours` method only, and only when `our_mode` is `soft`)
* `--ssm_warmup_method`: str, warmup method for the SSM, either `full` or `ssm`
* `--ssm_warmup_mode`: str, warmup mode for the SSM, either `old` or `new`  
* `--data`: str, dataset, either `mnist`, `cifar10`, or `imagenet`
* `--grayscale`: bool, whether to convert the dataset to grayscale
* `--pretrained_data`: str, pretrained dataset, either `mnist`, `cifar10`, or empty string (no pretrained data)


## Example Usage

Here's an example command to run an experiment using `run_s4.py`:

```bash
python run_s4.py \
    --n_epochs 500 \
    --batch_size 4000 \
    --seq_len 200 \
    --lr 0.005 \
    --wandb False \
    --warmup_epochs 20 \
    --warmup_lr 0.01 \
    --overwrite False \
    --d_model 64 \
    --frozen_n_layers 4 \
    --frozen_d_state 16 \
    --dropout 0.0 \
    --task_type regression \
    --lp_method ours \
    --D_method full \
    --ssm_method ours \
    --our_mode hard \
    --select_states_dim 8 \
    --updatable_states_dim 8 \
    --select_channels_dim 16 \
    --updatable_channels_dim 16 \
    --select_states_penalty 0.0 \
    --updatable_states_penalty 0.0 \
    --ssm_warmup_method full \
    --data synthetic \
    --target_n_layers 1 \
    --target_d_state 8 
```

This example demonstrates a typical configuration for running SDLoRA experiments on a synthetic dataset.