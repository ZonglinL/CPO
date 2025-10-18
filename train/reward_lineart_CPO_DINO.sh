
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export CONTROLNET_DIR="Your_finetuned_ControlNetDino"
export REWARDMODEL_DIR="https://huggingface.co/spaces/awacke1/Image-to-Line-Drawings/resolve/main/model.pth"
export OUTPUT_DIR="work_dirs/reward_model/MultiGen20M_LineDrawing/CPO_DINO"


accelerate launch --config_file "train/config.yml" \
 --main_process_port=27446 train/reward_CPO_DINO.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="lineart" \
 --dataset_name="data/MultiGen_Reward_DPO_cond/"  \
 --cache_dir="data/huggingface_datasets" \
 --caption_column="prompt" \
 --conditioning_image_column="lineart" \
 --resolution=512 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=8 \
 --learning_rate=3e-9 \
 --mixed_precision="fp16" \
 --dataloader_num_workers=8 \
 --max_train_steps=4000 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=1000 \
 --checkpointing_steps=1000 \
 --grad_scale=0.02 \
 --validation_steps=1000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200 \
 --margin=1e-2 \
 --max_val_samples=1 \
 --beta_dpo=5000 \