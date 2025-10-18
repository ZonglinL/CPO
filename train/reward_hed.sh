export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_softedge"
export REWARDMODEL_DIR="https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
export OUTPUT_DIR="work_dirs/reward_model/MultiGen20M_Hed/finetune_DINO"


accelerate launch --config_file "train/config.yml" \
 --main_process_port=27445 train/reward_DINO.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="hed" \
 --dataset_name="data/MultiGen_Reward_DPO_cond/"\
 --caption_column="prompt" \
 --conditioning_image_column="hed" \
 --resolution=512 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=8 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --dataloader_num_workers=4 \
 --max_train_steps=20000 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=1000 \
 --checkpointing_steps=1000 \
 --grad_scale=1 \
 --validation_steps=1000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200