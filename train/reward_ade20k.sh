export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_seg"
export REWARDMODEL_DIR="mmseg::upernet/upernet_r50_4xb4-160k_ade20k-512x512.py"
export OUTPUT_DIR="work_dirs/reward_model/Captioned_ADE20K/finetune_DINO"


# reward fine-tuning, you may train ControlNet++ for DINO-adapted ControlNet by changing to reward_control.py
accelerate launch --config_file "train/config.yml" \
 --main_process_port=23156 train/reward_DINO.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="segmentation" \
 --dataset_name="limingcv/Captioned_ADE20K" \
 --caption_column="prompt" \
 --conditioning_image_column="control_seg" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=8 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --dataloader_num_workers=4 \
 --max_train_steps=5000 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=1000 \
 --checkpointing_steps=1000 \
 --grad_scale=0.5 \
 --validation_steps=1000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200 \
