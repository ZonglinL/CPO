# Path to the controlnet weight (can be huggingface or local path)


## DPO
#export CONTROLNET_DIR="work_dirs/DPO/Captioned_COCOPose/reward-straight_controlnet_sd15_pose_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-1k_ema/"
#export CONTROLNET_DIR="work_dirs/DPO/Captioned_COCOPose/reward-straight_controlnet_sd15_pose_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-1k/checkpoint-4000/controlnet_ema"

## controlnet

export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_openpose"


#export CONTROLNET_DIR="work_dirs/controlnet/Captioned_COCOPose/controlnet_sd15_pose_res512_bs256_lr1e-5_warmup100_scale-0.5_iter10k_fp16_ema/"

## DPO time weight
#export CONTROLNET_DIR="work_dirs/DPO_time_weight_scale_1/Captioned_COCOPose/reward-straight_controlnet_sd15_pose_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-1k_ema/"


## controlnet++
#export CONTROLNET_DIR="work_dirs/reward_model/Captioned_COCOPose/reward_controlnet_sd15_pose_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_ema"

# Guidance scale and inference steps
export SCALE=7.5
export NUM_STEPS=50

# Generate images for evaluation
# If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.



#accelerate launch --config_file "eval/config.yml" --main_process_port=17777 eval/eval.py --task_name='pose' --dataset_name='limingcv/Captioned_COCOPose' --dataset_split='validation' --condition_column='control_pose' --label_column='pose' --prompt_column='prompt'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --ddim

accelerate launch --config_file "eval/config.yml" --num_processes=1 --main_process_port=12222 eval/eval_poses.py --task_name='pose' --dataset_name='limingcv/Captioned_COCOPose' --dataset_split='validation' --condition_column='control_pose' --label_column='pose' --prompt_column='prompt'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

#python eval/eval_fid.py --task_name='pose' --dataset_name='limingcv/Captioned_COCOPose' --dataset_split='validation' --condition_column='control_pose' --label_column='pose' --prompt_column='prompt'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}



#accelerate launch --config_file "eval/config.yml" --main_process_port=16666 eval/eval.py --task_name='pose' --dataset_name='limingcv/HumanArt' --dataset_split='validation' --condition_column='control_pose' --label_column='pose' --prompt_column='prompt'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --ddim

accelerate launch --config_file "eval/config.yml" --num_processes=1 --main_process_port=13333 eval/eval_poses.py --task_name='pose' --dataset_name='limingcv/HumanArt' --dataset_split='validation' --condition_column='control_pose' --label_column='pose' --prompt_column='prompt'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

#python eval/eval_fid.py --task_name='pose' --dataset_name='limingcv/HumanArt' --dataset_split='validation' --condition_column='control_pose' --label_column='pose' --prompt_column='prompt'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}