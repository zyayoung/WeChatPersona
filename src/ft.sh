export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

deepspeed src/ft.py  \
    --report_to "tensorboard" \
    --num_train_epochs 400 \
    --output_dir checkpoints/wechat_v3 \
    --optim adamw_torch \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_only_model True \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --lr_scheduler_type cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.01 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --bf16 True \
    --tf32 True
