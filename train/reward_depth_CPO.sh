
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export CONTROLNET_DIR="reward_controlnet_download/checkpoints/depth/controlnet"
export REWARDMODEL_DIR="Intel/dpt-hybrid-midas"
export OUTPUT_DIR="work_dirs/reward_model/MultiGen20M_Depth/CPO"


accelerate launch --config_file "train/config.yml" \
 --main_process_port=23152 train/reward_CPO.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="depth" \
 --dataset_name="data/MultiGen_Reward_DPO_cond/" \
 --keep_in_memory=False \
 --caption_column="prompt" \
 --conditioning_image_column="winner_depth" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=2 \
 --learning_rate=1e-8 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=8 \
 --max_train_steps=4000 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=100 \
 --checkpointing_steps=1000 \
 --grad_scale=0.05 \
 --use_ema \
 --validation_steps=1000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=600 \
 --beta_dpo=5000 \
 --max_val_samples=1 \
 --margin=1e-2 \
 --start_step=0 \
 #--use_8bit_adam