# pip3 install torchmetrics
import torch
import argparse

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.multimodal.clip_score import CLIPScore
import ImageReward as RM
import numpy as np


parser = argparse.ArgumentParser(description="Evaluate CLIP-Score")
parser.add_argument('--generated_image_dir', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True, default="limingcv/MultiGen-20M_canny_eval")
args = parser.parse_args()

dataset = load_dataset(args.dataset, cache_dir='data/huggingface_datasets', split='validation')
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").cuda()



bar = tqdm(range(len(dataset)), desc=f"Evaluating {args.dataset}")

rewards = []
for idx in range(len(dataset)):
    try:
        data = dataset[idx]
        if "MultiGen" in args.dataset:
            prompt = data["text"]
        else:
            prompt = data["prompt"]

        image_paths = [f'{args.generated_image_dir}/group_{i}/{idx}.png' for i in range(4)]
        images = [Image.open(x).convert('RGB') for x in image_paths]
        tensor_images = [pil_to_tensor(x).cuda() for x in images]
        metric.update(torch.stack(tensor_images), [prompt]*4)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        #continue ## pose validation data may not contain poses, which we did not save generated images --> skipped
    bar.update(1)
print(f"CLIP:{metric.score / metric.n_samples}")
