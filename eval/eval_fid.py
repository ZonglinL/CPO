# pip3 install clean-fid
# https://github.com/GaParmar/clean-fid
from cleanfid import fid
import argparse
import os

def main(real_image_path, generated_image_path):
    score = 0.0
    # We have 4 groups of generated images
    for i in range(4):
        score += fid.compute_fid(
            real_image_path,
            f'{generated_image_path}/group_{i}',
            dataset_res=512,
            batch_size=500
        )
    # Report the average FID score
    print(score / 4)

if __name__ == "__main__":
    # For real images, you should load our huggingface datasets and then save each image into local path.
    # For generated images, you should run our evaluate sctipts for each condition.
    # Make sure the real images and the generated images have the same file name.
    parser = argparse.ArgumentParser(description='Semantic Segmentation and Image Generation')
    parser.add_argument('--task_name', type=str, default='seg')
    parser.add_argument('--dataset_name', type=str, default='limingcv/Captioned_COCOStuff', help='Dataset name')
    parser.add_argument('--dataset_split', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='lllyasviel/control_v11p_sd15_seg')
    parser.add_argument('--sd_path', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--guidance_scale', type=float, default=12.5)
    parser.add_argument('--image_column', type=str, default='image')
    parser.add_argument('--condition_column', type=str, default='control_seg')
    parser.add_argument('--label_column', type=str, default=None)
    parser.add_argument('--prompt_column', type=str, default='prompt')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for image generation')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the image')
    parser.add_argument('--output_dir', type=str, default='work_dirs/eval_dirs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--cache_dir', type=str, default="data/huggingface_datasets", help='Cache directory for dataset and models')
    parser.add_argument('--model', type=str, default="controlnet")
    parser.add_argument('--ddim', action='store_true', help='weather use DDIM instead of UniPC')
    parser.add_argument('--num_inference_steps', type=int, default=20, help='Number of inference steps')

    args = parser.parse_args()


    save_dir = os.path.join(args.output_dir, args.dataset_name.split('/')[-1], args.dataset_split, args.model_path.replace('/', '_'))

    save_dir = save_dir + '_' + str(args.guidance_scale) + '-' + str(args.num_inference_steps)
    real_image_path = os.path.join(save_dir, "real_image")
    generated_image_path = os.path.join(save_dir, "images")

    main(real_image_path, generated_image_path)