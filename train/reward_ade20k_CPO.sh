export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export CONTROLNET_DIR="work_dirs/reward_model/Captioned_ADE20K/reward/checkpoint-5000/controlnet"
export REWARDMODEL_DIR="mmseg::upernet/upernet_r50_4xb4-160k_ade20k-512x512.py"
export OUTPUT_DIR="work_dirs/reward_model/Captioned_ADE20K/CPO"

accelerate launch --config_file "train/config.yml" \
 --main_process_port=23159 train/reward_CPO.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="segmentation" \
 --dataset_name="ucfzl/Segmentation_Reward_DPO_cond" \
 --keep_in_memory=False \
 --caption_column="prompt" \
 --conditioning_image_column="winner" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=2 \
 --learning_rate=1e-8 \
 --mixed_precision="fp16" \
 --dataloader_num_workers=4 \
 --max_train_steps=2000 \
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
 --start_step=0 \
 --margin=1e-2 \