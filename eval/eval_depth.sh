# Path to the controlnet weight (can be huggingface or local path)

export CONTROLNET_DIR="ucfzl/ControlNet_Depth_CPO"
#export CONTROLNET_DIR="ucfzl/ControlNet_DINO_Depth_CPO"
# How many GPUs and processes you want to use for evaluation.
export NUM_GPUS=2
# Guidance scale and inference steps
export SCALE=7.5
export NUM_STEPS=20

# Generate images for evaluation
# If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
accelerate launch --main_process_port=12356 --config_file "eval/config.yml" eval/eval_dino.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

# Path to the above generated images
# guidance_scale=7.5, sampling_steps=20 by default
export DATA_DIR="work_dirs/eval_dirs/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}


rm ${DATA_DIR}/images/*/*depth*

python eval/eval_fid.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

export IMAGE_DIR="${DATA_DIR}/images/"

python eval/eval_clip.py --generated_image_dir=$IMAGE_DIR --dataset='limingcv/MultiGen-20M_depth_eval'