accelerate launch --config_file configs/deepspeed/multi_gpu.yaml --num_processes 4 src/trl/sft.py \
    --model_name_or_path="gpt2" \
    --dataset_name=trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --report_to="wandb" \
    --learning_rate=5e-5 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --output_dir="sft_anthropic_hh" \
    --logging_steps=10 \
    --num_train_epochs=3 \
    --max_steps=-1