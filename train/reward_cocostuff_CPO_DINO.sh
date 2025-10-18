export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export CONTROLNET_DIR="Your_finetuned_ControlNetDino"
export REWARDMODEL_DIR="mmseg::deeplabv3/deeplabv3_r50-d8_4xb4-160k_coco-stuff164k-512x512.py"
export OUTPUT_DIR="work_dirs/reward_model/Captioned_COCOStuff/CPO_DINO"

# reward fine-tuning
accelerate launch --config_file "train/config.yml" \
 --main_process_port=23157 train/reward_CPO_DINO.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="segmentation" \
 --dataset_name="limingcv/Captioned_COCOStuff" \
 --caption_column="prompt" \
 --conditioning_image_column="control_seg" \
 --label_column="panoptic_seg_map" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=8 \
 --learning_rate=1e-8 \
 --mixed_precision="fp16" \
 --dataloader_num_workers=8 \
 --max_train_steps=2000 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=500 \
 --checkpointing_steps=500 \
 --grad_scale=0.05 \
 --validation_steps=500 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=600 \
 --margin=1e-2 \
 --max_val_samples=1 \