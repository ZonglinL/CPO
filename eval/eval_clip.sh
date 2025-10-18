# 'limingcv/Captioned_ADE20K'
# 'limingcv/Captioned_COCOPose'
# "limingcv/MultiGen-20M_canny_eval"
# "limingcv/MultiGen-20M_depth_eval"
# "limingcv/HumanArt"
# 'limingcv/Captioned_COCOStuff'
export IMAGE_DIR="/lustre/fs1/home/zlyu/Control-Anything/work_dirs/eval_dirs/MultiGen-20M_canny_eval/validation/work_dirs_DPO_MultiGen20M_LineDrawing_controlnet_sd15_ema_7.5-20/images"

python eval/eval_clip.py --generated_image_dir=$IMAGE_DIR --dataset="limingcv/MultiGen-20M_canny_eval"