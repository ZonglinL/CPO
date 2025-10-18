
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export CONTROLNET_DIR="/lustre/fs1/home/zo258499/Control-Anything/reward_controlnet_download/checkpoints/cocostuff/reward_5k"
export REWARDMODEL_DIR="mmseg::deeplabv3/deeplabv3_r50-d8_4xb4-160k_coco-stuff164k-512x512.py"
export OUTPUT_DIR="work_dirs/DPO/Captioned_COCOStuff/controlnet_sd15"


accelerate launch --config_file "train/config.yml" \
 --main_process_port=23156 train/reward_CPO.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="segmentation" \
 --dataset_name="ucfzl/COCOstuff" \
 --caption_column="prompt" \
 --conditioning_image_column="winner" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=8 \
 --learning_rate=1e-8 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=8 \
 --max_train_steps=10000 \
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
 --margin=1e-2 \
 --beta_dpo=5000 \
 --max_val_samples=1 \
 --start_step=0 \
