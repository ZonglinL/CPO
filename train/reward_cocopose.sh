export OUTPUT_DIR="work_dirs/reward_model/Captioned_COCOPose/finetune_DINO"
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_openpose"
export REWARDMODEL_DIR="mmpose::body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512.py"

## you will need to finetune an original ControlNet version of cocopose by using reward_control_original 

accelerate launch --config_file "train/config.yml" \
 --main_process_port=23156 train/reward_DINO.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="pose" \
 --dataset_name="limingcv/Captioned_COCOPose"  \
 --keep_in_memory=False \
 --caption_column="prompt" \
 --conditioning_image_column="control_pose" \
 --label_column="pose" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=14 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --dataloader_num_workers=4 \
 --max_train_steps=20000 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=10 \
 --checkpointing_steps=1000 \
 --grad_scale=0.5 \
 --validation_steps=1000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200\
 --max_val_samples=1