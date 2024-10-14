#!/bin/bash

# ====== Select Dimensions ======
# magnitude: 531 out of 576 (top 92.22%)
# loss: 175 out of 576 (top 30.38%)
python run_s4.py \
    --d_model 4 \
    --frozen_n_layers 2 \
    --frozen_d_state 2 \
    --target_n_layers 1 \
    --target_d_state 1 \
    --select_states_dim 1 \
    --updatable_states_dim 1 \
    --select_channels_dim 2 \
    --updatable_channels_dim 2 \
    --select_dim_only True \
    --ssm_method ours \
    --lp_method ours \
    --wandb False \
    --pretrain True \
    --pretrain_epochs 50 \
    --ssm_warmup_mode loss


# ====== Update Dimensions ======
# magnitude: 137 out of 180 (top 76.11%)
# loss: 43 out of 180 (top 23.89%)
# grad: 167 out of 180 (top 92.78%)
python run_s4.py \
    --d_model 4 \
    --frozen_n_layers 1 \
    --frozen_d_state 4 \
    --target_n_layers 1 \
    --target_d_state 4 \
    --select_states_dim 4 \
    --updatable_states_dim 2 \
    --select_channels_dim 4 \
    --updatable_channels_dim 2 \
    --update_dim_only True \
    --ssm_method ours \
    --lp_method ours \
    --wandb False \
    --pretrain True \
    --pretrain_epochs 50 \
    --ssm_warmup_mode grad