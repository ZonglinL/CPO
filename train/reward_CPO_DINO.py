#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import os
import math
import time
import kornia
import pickle
import random
import logging
import argparse

from itertools import product
import torch
import numpy as np
import accelerate
import transformers
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk
from huggingface_hub import create_repo, upload_folder
from transformers import AutoTokenizer, PretrainedConfig,UperNetForSemanticSegmentation

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from packaging import version
from torchvision import transforms
from torch.cuda.amp import autocast
from torchvision.transforms.functional import normalize
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.Resampling.BICUBIC
import cv2

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from torchvision.ops import nms,box_iou

import diffusers
from diffusers import (
    AutoencoderKL,
    #ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controlnets.controlnet import ControlNetModel


from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from utils import image_grid, get_reward_model, get_reward_loss, label_transform, group_random_crop


from PIL import PngImagePlugin
MaximumDecompressedsize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedsize * MegaByte
Image.MAX_IMAGE_PIXELS = None


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__)

# Offloading state_dict to CPU to avoid GPU memory boom (only used for FSDP training)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

from transformers import DPTForDepthEstimation
import io, gc
def jpeg_smooth_tensor(depth: torch.Tensor,
                       quality: int = 75,
                       clear_cuda: bool = False) -> torch.Tensor:
    """
    Apply JPEG smoothing, freeing memory after each image.

    Args:
      depth      : (B, H, W) float tensor in [0,1].
      quality    : JPEG quality [1–100].
      clear_cuda : if True, runs torch.cuda.empty_cache() at the end.
    """
    device = depth.device
    dtype = depth.dtype
    out_list = []

    for img in depth.detach().cpu():
        # 1) Convert to PIL in a local scope
        arr = (img.numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode='L')

        # 2) JPEG round-trip inside context managers
        with io.BytesIO() as buf:
            pil.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            with Image.open(buf) as comp:
                comp_arr = np.array(comp, dtype=np.float32) / 255.0

        # 3) Append result and delete intermediates immediately
        out_list.append(torch.from_numpy(comp_arr))
        del arr, pil, comp, comp_arr
        gc.collect()

    # 4) Stack and move to device
    out = torch.stack(out_list, dim=0).to(device = device,dtype = dtype)

    # 5) Optionally clear CUDA cache and run GC again
    if clear_cuda and device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return out

def filter_yolo_pose_results(result,
                              conf_threshold=0.2,
                              iou_threshold=0.9):
    """
    Filter and normalize YOLO pose estimation results by confidence and IoU.
    Returns bboxes (xywh), keypoints (xy), and visibility, all normalized.
    """

    boxes = result.boxes
    keypoints_raw = result.keypoints

    # Early exit if no detections
    if boxes is None or boxes.conf.numel() == 0 or keypoints_raw is None:
        return (
            torch.empty((0, 4)),     # bboxes_xywh
            torch.empty((0, 0, 2)),  # keypoints
            torch.empty((0, 0))      # visibility
        )

    confs = boxes.conf  # (N,)
    bboxes_xyxy = boxes.xyxy  # (N, 4)
    bboxes_xywh = boxes.xywh  # (N, 4)

    keypoints = keypoints_raw.xy  # (N, K, 2)
    visibility = keypoints_raw.conf  # (N, K)

    # Step 1: Filter by confidence
    mask = confs > conf_threshold
    if mask.sum() == 0:
        return (
            torch.empty((0, 4)),
            torch.empty((0, keypoints.shape[1], 2)),
            torch.empty((0, keypoints.shape[1]))
        )

    bboxes_xywh = bboxes_xywh[mask]
    bboxes_xyxy = bboxes_xyxy[mask]
    keypoints = keypoints[mask]
    visibility = visibility[mask]

    # Step 2: IoU filtering
    ious = box_iou(bboxes_xyxy, bboxes_xyxy)
    keep = []
    for i in range(len(ious)):
        if all(ious[i, keep] < iou_threshold if keep else [True]):
            keep.append(i)

    bboxes_xywh = bboxes_xywh[keep]
    keypoints = keypoints[keep]
    visibility = visibility[keep]

    return bboxes_xywh, keypoints, visibility



limb_seq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
            [1, 16], [16, 18]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255,
                                                     0], [170, 255, 0],
          [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
          [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0,
                                                    255], [255, 0, 255],
          [255, 0, 170], [255, 0, 85]]

stickwidth = 4
num_openpose_kpt = 18
num_link = len(limb_seq)

def compute_bbox_areas(bboxes):
    """
    Compute the area for a set of bounding boxes.

    bboxes: (n, 4) array where each row is (x_min, y_min, x_max, y_max)
    Returns: (n,) array of bounding box areas
    """
    bboxes[:,:2] -= bboxes[:,2:] / 2
    bbox_w, bbox_h = bboxes[:,2],bboxes[:,3]

    # Compute area
    areas = bbox_w * bbox_h
    return areas


def mmpose_to_openpose_visualization_pil(img_pil, keypoints, kpt_thr=0.5):
    """Visualize predicted keypoints of one image in openpose format using a PIL image."""
    
    # Convert PIL image to NumPy array
    img = np.array(img_pil)
    black_img = np.zeros_like(img)
    # compute neck joint    
    # # First, ensure keypoints is a tensor and has at least 2 dimensions
    if keypoints.ndim < 2:
        img_pil = Image.fromarray(black_img.astype(np.uint8))
        return img_pil

    # Check if the second dimension (axis 1) has size 0
    # This means no keypoints were detected for any person, or no people were detected at all
    if keypoints.shape[1] == 0:
        img_pil = Image.fromarray(black_img.astype(np.uint8))
        return img_pil

    neck = (keypoints[:, 5] + keypoints[:, 6]) / 2


    if np.any(keypoints[:, 5, 2]) < kpt_thr or np.any(keypoints[:, 6, 2]) < kpt_thr:
        neck[:, 2] = 0

    # 17 keypoints to 18 keypoints
    new_keypoints = np.insert(keypoints[:, ], 17, neck, axis=1)

    # mmpose format to openpose format
    openpose_idx = [15, 14, 17, 16, 2, 6, 3, 7, 4, 8, 12, 9, 13, 10, 1]
    mmpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints[:, openpose_idx, :] = new_keypoints[:, mmpose_idx, :]

    # black background
    

    num_instance = new_keypoints.shape[0]

    # draw keypoints
    for i, j in product(range(num_instance), range(num_openpose_kpt)):
        x, y, conf = new_keypoints[i][j]
        if conf > kpt_thr:
            cv2.circle(black_img, (int(x), int(y)), 4, colors[j], thickness=-1)

    # draw links
    cur_black_img = black_img.copy()
    for i, link_idx in product(range(num_instance), range(num_link)):
        conf = new_keypoints[i][np.array(limb_seq[link_idx]) - 1, 2]
        if np.sum(conf > kpt_thr) == 2:
            Y = new_keypoints[i][np.array(limb_seq[link_idx]) - 1, 0]
            X = new_keypoints[i][np.array(limb_seq[link_idx]) - 1, 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle),
                0, 360, 1)
            cv2.fillConvexPoly(cur_black_img, polygon, colors[link_idx])
    black_img = cv2.addWeighted(black_img, 0.4, cur_black_img, 0.6, 0)

    # Convert NumPy array back to PIL image
    img_pil = Image.fromarray(black_img.astype(np.uint8))
    return img_pil


def log_validation(
        vae,
        text_encoder,
        tokenizer,
        unet,
        controlnet,
        ema_controlnet,
        args,
        accelerator,
        weight_dtype,
        step,
        val_dataset):

    # randomly select some samples to log
    if val_dataset is not None:
        val_dataset = val_dataset.select(
            random.sample(range(len(val_dataset)), args.max_val_samples)
        )

    if args.task_name in ['lineart', 'hed']:
        reward_model = get_reward_model(args.task_name, args.reward_model_name_or_path)
        reward_model.to(device = accelerator.device, dtype = weight_dtype)
        reward_model.eval()
    elif args.task_name == 'depth':
        reward_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        reward_model.to(accelerator.device)
        reward_model.eval()
    else:
        reward_model = None

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_column = args.image_column
    caption_column = args.caption_column

    if args.conditioning_image_column in ['canny', 'lineart', 'hed','depth']:
        conditioning_image_column = image_column
    else:
        conditioning_image_column = args.conditioning_image_column

    assert val_dataset is not None, "Validation dataset is required for logging validation images."
    try:
        validation_images = val_dataset[image_column]
        validation_conditions = val_dataset[conditioning_image_column]
        validation_prompts = val_dataset[caption_column]
    except:
        validation_images = [item[image_column] for item in val_dataset]
        validation_conditions = [item[conditioning_image_column] for item in val_dataset]
        validation_prompts = [item[caption_column] for item in val_dataset]

    # Avoid some problems caused by grayscale images
    validation_conditions = [x.convert('RGB') for x in validation_conditions]

    if args.conditioning_image_column == "canny":
        low_threshold = 0.15 # low_threshold = random.uniform(0, 1)
        high_threshold = 0.3 # high_threshold = random.uniform(low_threshold, 1)
        with autocast():
            validation_conditions = [torchvision.transforms.functional.pil_to_tensor(x) for x in validation_conditions]
            validation_conditions = [x / 255. for x in validation_conditions]
            validation_conditions = kornia.filters.canny(torch.stack(validation_conditions), low_threshold, high_threshold)[1]
            validation_conditions = torch.chunk(validation_conditions, len(validation_conditions), dim=0)
            validation_conditions = [torchvision.transforms.functional.to_pil_image(x.squeeze(0), 'L') for x in validation_conditions]
    elif args.conditioning_image_column in ['lineart', 'hed']:
        with autocast():
            validation_conditions = [torchvision.transforms.functional.pil_to_tensor(x) for x in validation_conditions]
            validation_conditions = [x / 255. for x in validation_conditions]
            validation_conditions = [torchvision.transforms.functional.resize(x, (512, 512), interpolation=Image.BICUBIC) for x in validation_conditions]
            with torch.no_grad():
                validation_conditions = reward_model(torch.stack(validation_conditions).to(accelerator.device))
            validation_conditions = 1 - validation_conditions if args.task_name == 'lineart' else validation_conditions
            validation_conditions = torch.chunk(validation_conditions, len(validation_conditions), dim=0)
            validation_conditions = [torchvision.transforms.functional.to_pil_image(x.squeeze(0), 'L') for x in validation_conditions]
    elif args.conditioning_image_column == 'depth':
        target_short_side = 384
        with autocast():
            with torch.no_grad():
                # mean & std used in image transformations

                validation_conditions = [torchvision.transforms.functional.pil_to_tensor(x) for x in validation_conditions]
                validation_conditions = [x / 255. for x in validation_conditions]
                validation_conditions = torch.stack(validation_conditions).to(accelerator.device) ## 0,1

                num_val = validation_conditions.shape[0]


                height,width = validation_conditions.shape[-2:] ## h w
                if width < height: # Width is the short side
                    new_width = target_short_side
                    new_height = int(height * (new_width / width))
                else: # Height is the short side (or equal)
                    new_height = target_short_side
                    new_width = int(width * (new_height / height))

                depth_input = torchvision.transforms.functional.resize(validation_conditions, size=(new_height, new_width), interpolation=transforms.InterpolationMode.BICUBIC)

                depth_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                depth_std = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                #depth_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(accelerator.device)
                #depth_std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(accelerator.device)
                
                depth_input = (depth_input - depth_mean)/depth_std

                outputs = reward_model(depth_input.to(weight_dtype)) ## B H W

                outputs = outputs.predicted_depth ## B H W
                outputs = torchvision.transforms.functional.resize(outputs, (height, width), interpolation=transforms.InterpolationMode.BICUBIC)
                min_values = outputs.view(num_val, -1).amin(dim=1, keepdim=True).view(num_val, 1, 1)
                outputs = outputs - min_values
                max_values = outputs.view(num_val, -1).amax(dim=1, keepdim=True).view(num_val, 1, 1)

                outputs = outputs / max_values ## 0,1

                outputs = jpeg_smooth_tensor(outputs)

                validation_conditions = outputs

                validation_conditions = torch.chunk(validation_conditions, len(validation_conditions), dim=0)
                validation_conditions = [torchvision.transforms.functional.to_pil_image(x.squeeze(0), 'L').convert('RGB') for x in validation_conditions] ## h w to gray
    image_logs = []

    logger.info(f"Running validation with {len(validation_prompts)} prompts... ")
    for validation_prompt, validation_condition, validation_image in zip(validation_prompts, validation_conditions, validation_images):
        if val_dataset is not None:
            validation_image = validation_image.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
            validation_condition = validation_condition.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
        else:
            validation_condition = Image.open(validation_condition).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)

        with torch.autocast("cuda"):
            images = pipeline(
                [validation_prompt] * args.num_validation_images,
                [validation_condition] * args.num_validation_images,
                num_inference_steps=20,
                generator=generator
            ).images

        image_logs.append({
            "validation_image": validation_image,
            "validation_condition": validation_condition,
            "validation_prompt": validation_prompt,
            "images": images,
            'ema_images': []
        })

    if args.use_ema:
        # Store the ControlNet parameters temporarily and load the EMA parameters to perform inference.
        ema_controlnet.copy_to(controlnet.parameters())

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            safety_checker=None,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        logger.info(f"Running validation with {len(validation_prompts)} prompts... ")
        for idx, (validation_prompt, validation_condition, validation_image) in enumerate(zip(validation_prompts, validation_conditions, validation_images)):
            if val_dataset is not None:
                validation_condition = validation_condition.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
            else:
                validation_condition = Image.open(validation_condition).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)

            with torch.autocast("cuda"):
                images = pipeline(
                    [validation_prompt] * args.num_validation_images,
                    [validation_condition] * args.num_validation_images,
                    num_inference_steps=20,
                    generator=generator
                ).images

            image_logs[idx]['ema_images'] = images


    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                ema_images = log["ema_images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                validation_condition = log["validation_condition"]

                validation_prompt = validation_prompt + ['EMA'] * len(validation_prompt)

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                for image in ema_images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                images = log["images"]
                ema_images = log["ema_images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                validation_condition = log["validation_condition"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet input image"))
                formatted_images.append(wandb.Image(validation_condition, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

                for image in ema_images:
                    image = wandb.Image(image, caption='EMA')
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        if reward_model is not None:
            reward_model = None

        return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--reward_model_name_or_path",
        type=str,
        default=None,
        help="Path to reward model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--grad_scale", type=float, default=1, help="Scale divided for grad loss value."
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--control_guidance_start",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--control_guidance_end",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--beta_dpo", type=float, default=5000, help="Beta scaler for DPO. this controls step size."
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default='limingcv/reward_controlnet',
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--timestep_sampling_start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--timestep_sampling_end",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--min_timestep_rewarding",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_timestep_rewarding",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default='segmentation',
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--keep_in_memory",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--wrong_ids_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="The column of the dataset containing the original labels. `seg_map` for ADE20K; `panoptic_seg_map` for COCO-Stuff.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=1,
        help=(
            "Max number of samples for validation during training, default to 10"
        ),
    )
    parser.add_argument(
        "--image_condition_dropout",
        type=float,
        default=0,
        help="Probability of image conditions to be replaced with tensors with zero value. Defaults to 0.",
    )
    parser.add_argument(
        "--text_condition_dropout",
        type=float,
        default=0,
        help="Probability of image prompts to be replaced with empty strings. Defaults to 0.05.",
    )
    parser.add_argument(
        "--all_condition_dropout",
        type=float,
        default=0,
        help="Probability of abandon all the conditions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="reward_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=2e-3,
        help="start step of DPO",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.text_condition_dropout < 0 or args.text_condition_dropout > 1:
        raise ValueError("`--text_condition_dropout` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args 


def make_train_dataset(args, tokenizer, accelerator, split='train', codec=None):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        if args.dataset_name.count('/') == 1:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                keep_in_memory=args.keep_in_memory,
            )
        else:
            dataset = load_from_disk(
                dataset_path=args.dataset_name,
                keep_in_memory=args.keep_in_memory,
            )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
                keep_in_memory=args.keep_in_memory,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    try:
        column_names = dataset[split].column_names
    except:
        column_names = dataset.features

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    elif args.conditioning_image_column in ['canny', 'lineart', 'hed','depth']:
        conditioning_image_column = image_column
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    resolution = (args.resolution, args.resolution)
    image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    label_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST, antialias=True),
            # transforms.CenterCrop(args.resolution),
        ]
    )

    def preprocess_train(examples):
        pil_images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in pil_images]

        if args.conditioning_image_column in ['canny', 'lineart', 'hed','depth']:
            conditioning_images = images
        else:
            conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
            conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        if args.label_column is not None:
            if args.task_name == "pose":
                labels =[np.array(label) for label in examples[args.label_column]]
                poses = [label[:, :, :2] for label in labels]
                poses_vis = [label[:, :, 2] for label in labels]

                input_image_sizes = [np.array(x.size) for x in pil_images]               # width, height
                output_imgae_sizes = [np.flip(np.array(x.shape[-2:])) for x in images]   # width, height

                poses = [pose*output_size/input_size for pose,input_size,output_size in zip(poses,input_image_sizes,output_imgae_sizes)]

                labels = [codec.encode(poses[i], poses_vis[i]) for i in range(len(labels))]
                labels = [torch.cat([torch.from_numpy(v) for _, v in x.items()]) for x in labels]

                examples[args.label_column] = labels
            else:
                dtype = torch.long
                labels = [torch.tensor(np.asarray(label), dtype=dtype).unsqueeze(0) for label in examples[args.label_column]]
                labels = [label_image_transforms(label) for label in labels]

        # perform groupped random crop for image/conditioning_image/label
        if args.label_column is not None and args.task_name != "pose":
            grouped_data = [torch.cat([x, y, z]) for (x, y, z) in zip(images, conditioning_images, labels)]
            grouped_data = group_random_crop(grouped_data, args.resolution)

            images = [x[:3, :, :] for x in grouped_data]
            conditioning_images = [x[3:6, :, :] for x in grouped_data]
            labels = [x[6:, :, :] for x in grouped_data]

            # (1, H, W) => (H, w)
            if args.task_name == "segmentation":
                labels = [label.squeeze(0) for label in labels]

            examples[args.label_column] = labels
        else:
            grouped_data = [torch.cat([x, y]) for (x, y) in zip(images, conditioning_images)]
            grouped_data = group_random_crop(grouped_data, args.resolution)

            images = [x[:3, :, :] for x in grouped_data]
            conditioning_images = [x[3:, :, :] for x in grouped_data]

        # Dropout some of features for classifier-free guidance.
        for i, img_condition in enumerate(conditioning_images):
            rand_num = random.random()
            if rand_num < args.image_condition_dropout:
                conditioning_images[i] = torch.zeros_like(img_condition)
            elif rand_num < args.image_condition_dropout + args.text_condition_dropout:
                examples[caption_column][i] = ""
            elif rand_num < args.image_condition_dropout + args.text_condition_dropout + args.all_condition_dropout:
                conditioning_images[i] = torch.zeros_like(img_condition)
                examples[caption_column][i] = ""

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        try:        
            if args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed)
                # rewrite the shuffled dataset on disk as contiguous chunks of data
                dataset["train"] = dataset["train"].flatten_indices()
                dataset["train"] = dataset["train"].select(range(args.max_train_samples))

            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)
        except:
            if args.max_train_samples is not None:
                dataset = dataset.shuffle(seed=args.seed)
                # rewrite the shuffled dataset on disk as contiguous chunks of data
                dataset= dataset.flatten_indices()
                dataset = dataset.select(range(args.max_train_samples))

            # Set the training transforms
            train_dataset = dataset.with_transform(preprocess_train)

    return dataset, train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    if args.label_column is not None:
        labels = torch.stack([example[args.label_column] for example in examples])
        labels = labels.to(memory_format=torch.contiguous_format).float()
    else:
        labels = conditioning_pixel_values

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "labels": labels,
    }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )


    
    if 'ADE' in args.dataset_name:
        reward_model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-large")
    elif args.task_name == 'depth':
        reward_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    else:
        reward_model = get_reward_model(args.task_name, args.reward_model_name_or_path)
    

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        cloned_controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        #controlnet = ControlNetModel.from_ControlNet(args.controlnet_model_name_or_path)
        #cloned_controlnet = ControlNetModel.from_ControlNet(args.controlnet_model_name_or_path)
        ref_controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)
        cloned_controlnet = ControlNetModel.from_unet(unet)
        ref_controlnet = ControlNetModel.from_ControlNet(args.controlnet_model_name_or_path)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "controlnet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    reward_model.requires_grad_(False)
    cloned_controlnet.requires_grad_(False)
    ref_controlnet.requires_grad_(False)
    controlnet.train()
    controlnet.config["cross_attention_dim"] = 768

    # Create EMA for the ControlNet.
    if args.use_ema:
        ema_controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        ema_controlnet = EMAModel(ema_controlnet.parameters(), model_cls=ControlNetModel, model_config=ema_controlnet.config)
    else:
        ema_controlnet = None

    # pose <-> heatmap transformation
    if args.task_name == "pose":
        from mmpose.codecs import SPR
        codec = SPR(
            input_size=(args.resolution, args.resolution),
            heatmap_size=(args.resolution//4, args.resolution//4),
            sigma=(4, 2),
            minimal_diagonal_length=32**0.5,
            generate_keypoint_heatmaps=True,
            decode_max_instances=30
        )
    else:
        codec = None

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if args.use_ema:
        ema_controlnet.to(accelerator.device)

    # Optimizer creation
    # optimized_parameters = list(controlnet.parameters()) + list(reward_model.parameters()) + list(unet.parameters())
    optimizer = optimizer_class(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset, train_dataset = make_train_dataset(args, tokenizer, accelerator, codec=codec)

    if args.validation_prompt is None and args.validation_image is None:
        try:
            val_dataset = dataset['validation']
        except:
            dataset = train_dataset.train_test_split(test_size=0.00005)
            train_dataset, val_dataset = dataset['train'], dataset['test']
    else:
        val_dataset = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # unet, reward_model = accelerator.prepare(unet, reward_model)

    # Prepare others after preparing the model
    controlnet, ref_controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet,ref_controlnet, optimizer, train_dataloader, lr_scheduler
    )


    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    reward_model.to(accelerator.device, dtype=weight_dtype)
    reward_model.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")
        from wandb import Settings
        wandb_settings = Settings(init_timeout=300)
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.output_dir.split('/')[-1]}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.resume_from_checkpoint))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=cloned_controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.task_name == 'segmentation':
        if 'ADE' in args.dataset_name:

            classes=('background', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
                    'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
                    'person', 'earth', 'door', 'table', 'mountain', 'plant',
                    'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
                    'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
                    'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                    'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                    'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                    'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
                    'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
                    'screen door', 'stairway', 'river', 'bridge', 'bookcase',
                    'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
                    'bench', 'countertop', 'stove', 'palm', 'kitchen island',
                    'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
                    'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                    'chandelier', 'awning', 'streetlight', 'booth',
                    'television receiver', 'airplane', 'dirt track', 'apparel',
                    'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
                    'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
                    'conveyer belt', 'canopy', 'washer', 'plaything',
                    'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
                    'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                    'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
                    'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
                    'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
                    'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
                    'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                    'clock', 'flag')


            palette=[[0, 0, 0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                    [102, 255, 0], [92, 0, 255]]
        else:
            classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
            palette = [[220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30], [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157], [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240], [0, 125, 92], [209, 0, 151], [188, 208, 182], [0, 220, 176], [255, 99, 164], [92, 0, 73], [133, 129, 255], [78, 180, 255], [0, 228, 0], [174, 255, 243], [45, 89, 255], [134, 134, 103], [145, 148, 174], [255, 208, 186], [197, 226, 255], [171, 134, 1], [109, 63, 54], [207, 138, 255], [151, 0, 95], [9, 80, 61], [84, 105, 51], [74, 65, 105], [166, 196, 102], [208, 195, 210], [255, 109, 65], [0, 143, 149], [179, 0, 194], [209, 99, 106], [5, 121, 0], [227, 255, 205], [147, 186, 208], [153, 69, 1], [3, 95, 161], [163, 255, 0], [119, 0, 170], [0, 182, 199], [0, 165, 120], [183, 130, 88], [95, 32, 0], [130, 114, 135], [110, 129, 133], [166, 74, 118], [219, 142, 185], [79, 210, 114], [178, 90, 62], [65, 70, 15], [127, 167, 115], [59, 105, 106], [142, 108, 45], [196, 172, 0], [95, 54, 80], [128, 76, 255], [201, 57, 1], [246, 0, 122], [191, 162, 208], [255, 255, 128], [147, 211, 203], [150, 100, 100], [168, 171, 172], [146, 112, 198], [210, 170, 100], [92, 136, 89], [218, 88, 184], [241, 129, 0], [217, 17, 255], [124, 74, 181], [70, 70, 70], [255, 228, 255], [154, 208, 0], [193, 0, 92], [76, 91, 113], [255, 180, 195], [106, 154, 176], [230, 150, 140], [60, 143, 255], [128, 64, 128], [92, 82, 55], [254, 212, 124], [73, 77, 174], [255, 160, 98], [255, 255, 255], [104, 84, 109], [169, 164, 131], [225, 199, 255], [137, 54, 74], [135, 158, 223], [7, 246, 231], [107, 255, 200], [58, 41, 149], [183, 121, 142], [255, 73, 97], [107, 142, 35], [190, 153, 153], [146, 139, 141], [70, 130, 180], [134, 199, 156], [209, 226, 140], [96, 36, 108], [96, 96, 96], [64, 170, 64], [152, 251, 152], [208, 229, 228], [206, 186, 171], [152, 161, 64], [116, 112, 0], [0, 114, 143], [102, 102, 156], [250, 141, 255]]

            #coco_processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-coco")
            #coco_model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")
    for epoch in range(first_epoch, args.num_train_epochs):
        loss_per_epoch = 0.
        pretrain_loss_per_epoch = 0.
        reward_loss_per_epoch = 0.

        train_loss, train_pretrain_loss, train_reward_loss = 0., 0., 0.

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]  # text condition
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)  # image condition

                # This step is necessary. It took us a long time to find out this issue
                # The input of the canny/hed/lineart model does not require normalization of the image
                if args.conditioning_image_column == "canny":
                    low_threshold = 0.15 # low_threshold = random.uniform(0, 1)
                    high_threshold = 0.3 # high_threshold = random.uniform(low_threshold, 1)
                    with torch.no_grad():
                        # mean & std used in image transformations
                        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        std = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        # magnitude, edge
                        denormalized_condition_image = controlnet_image * std + mean
                        labels, controlnet_image = reward_model(denormalized_condition_image, low_threshold, high_threshold)
                        controlnet_image = controlnet_image.expand(-1, 3, -1, -1)  # (B, 3, H, W)
                elif args.conditioning_image_column in ['lineart', 'hed']:
                    with torch.no_grad():
                        # mean & std used in image transformations
                        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        std = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        denormalized_condition_image = controlnet_image * std + mean
                        labels = reward_model(denormalized_condition_image.to(weight_dtype))
                        controlnet_image = labels.expand(-1, 3, -1, -1)  # (B, 3, H, W)
                        controlnet_image = 1 - controlnet_image if args.task_name == 'lineart' else controlnet_image

                elif args.conditioning_image_column == 'depth':
                    target_short_side = 384
                    with torch.no_grad():
                        # mean & std used in image transformations
                        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        std = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        denormalized_condition_image = controlnet_image * std + mean
                        height,width = denormalized_condition_image.shape[-2:] ## h w
                        if width < height: # Width is the short side
                            new_width = target_short_side
                            new_height = int(height * (new_width / width))
                        else: # Height is the short side (or equal)
                            new_height = target_short_side
                            new_width = int(width * (new_height / height))
                        depth_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        depth_std = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        #depth_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(accelerator.device)
                        #depth_std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(accelerator.device)                   
                        
                        depth_input = (denormalized_condition_image - depth_mean)/depth_std
                        #depth_input = denormalized_condition_image

                        depth_input = torchvision.transforms.functional.resize(depth_input, size=(new_height, new_width), interpolation=transforms.InterpolationMode.BICUBIC)

                        outputs = reward_model(depth_input.to(weight_dtype)) ## B H W
                        
                        outputs = outputs.predicted_depth ## B H W                  
                        min_values = outputs.view(args.train_batch_size, -1).amin(dim=1, keepdim=True).view(args.train_batch_size, 1, 1)
                        outputs = outputs - min_values

                        max_values = outputs.view(args.train_batch_size, -1).amax(dim=1, keepdim=True).view(args.train_batch_size, 1, 1)
                        outputs = outputs / max_values ## 0,1   B H W
                        labels = outputs.detach().clone() 
                        
                        outputs = torchvision.transforms.functional.resize(outputs, (height, width), interpolation=transforms.InterpolationMode.BICUBIC) ## args.resolution for conditioning
                        outputs = jpeg_smooth_tensor(outputs)
                        controlnet_image = outputs.unsqueeze(1).expand(-1, 3, -1, -1).detach().clone()  # (B, 3, H, W)
                        
                conds = [torchvision.transforms.functional.to_pil_image(x).convert('RGB') for x in controlnet_image] ## bchw -- list of pil
                prompts = tokenizer.batch_decode(batch['input_ids'])
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        sampled_images = pipeline(
                            prompts,
                            conds,
                            num_inference_steps=20
                        ).images ## list of sampled images
                
                
                losers = []
                image = torch.stack([torchvision.transforms.functional.pil_to_tensor(im)/255 for im in sampled_images]).to(device = accelerator.device,dtype = weight_dtype)
                if args.task_name == 'pose':
                    reward_model.to(torch.float32)
                    for im in image:
                        with torch.no_grad():
                            results = reward_model(image.to(torch.float32), verbose=False)[0]
                        boxes_filt,keypoints_filt,vis = filter_yolo_pose_results(results)
                        vis = (vis > 0.5).float()
                        scales = compute_bbox_areas(boxes_filt)#*args.resolution
                        mask = (vis != 0).any(-1)
                        keypoints_filt = keypoints_filt[mask]
                        vis = vis[mask]
                        label = torch.cat((keypoints_filt,vis.unsqueeze(-1)),-1)## key points *args.resolution
                        condition = mmpose_to_openpose_visualization_pil(torchvision.transforms.functional.to_pil_image(im.cpu()),label.cpu().numpy())
                        losers.append(torchvision.transforms.functional.pil_to_tensor(condition)/255.) 
                    losers = torch.stack(losers).to(device = accelerator.device,dtype = weight_dtype)

                else:

                    if args.task_name == "canny":
                        low_threshold = 0.15 # low_threshold = random.uniform(0, 1)
                        high_threshold = 0.3 # high_threshold = random.uniform(low_threshold, 1)
                        with torch.no_grad():
                            _, losers = reward_model(image, low_threshold, high_threshold)
                            losers = losers.expand(-1, 3, -1, -1)  # (B, 3, H, W)
                    
                    elif args.task_name in ['lineart', 'hed']:
                        with torch.no_grad():
                            losers = reward_model(image.to(weight_dtype))
                            losers = losers.expand(-1, 3, -1, -1)  # (B, 3, H, W)
                            losers = 1 - losers if args.task_name == 'lineart' else losers
                    elif args.task_name == 'depth':
                        target_short_side = 384

                        height,width = image.shape[-2:] ## h w
                        if width < height: # Width is the short side
                            new_width = target_short_side
                            new_height = int(height * (new_width / width))
                        else: # Height is the short side (or equal)
                            new_height = target_short_side
                            new_width = int(width * (new_height / height))
                        depth_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        depth_std = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        
                        #depth_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(accelerator.device)
                        #depth_std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(accelerator.device)  

                        depth_input = (image - depth_mean)/depth_std

                        depth_input = torchvision.transforms.functional.resize(depth_input, size=(new_height, new_width), interpolation=transforms.InterpolationMode.BICUBIC)

                        with torch.no_grad():
                            outputs = reward_model(depth_input.to(weight_dtype)) ## B H W
                        
                        outputs = outputs.predicted_depth ## B H W                  
                        min_values = outputs.view(args.train_batch_size, -1).amin(dim=1, keepdim=True).view(args.train_batch_size, 1, 1)
                        outputs = outputs - min_values

                        max_values = outputs.view(args.train_batch_size, -1).amax(dim=1, keepdim=True).view(args.train_batch_size, 1, 1)
                        outputs = outputs / max_values ## 0,1
                        outputs = torchvision.transforms.functional.resize(outputs, (height, width), interpolation=transforms.InterpolationMode.BICUBIC) ## args.resolution for conditioning
                        outputs = jpeg_smooth_tensor(outputs)

                        losers = outputs.unsqueeze(1).expand(-1, 3, -1, -1).detach().clone()  # (B, 3, H, W)

                    elif args.task_name =='segmentation':
                        if 'ADE' in args.dataset_name:
                            with torch.no_grad():
                                image = normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                outputs = reward_model(image.to(accelerator.device)).logits
                                seg_maps = outputs.argmax(1).cpu().numpy() + 1 ## B H W

                        else:
                            processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-coco")
                            model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")

                            model.to(accelerator.device)
                            model.eval()
                            inputs = processor(images=sampled_images, return_tensors="pt")
                            inputs['pixel_values'] = inputs['pixel_values'].to(accelerator.device)

                            with torch.no_grad():
                                outputs = model(**inputs)
                            class_queries_logits = outputs.class_queries_logits
                            masks_queries_logits = outputs.masks_queries_logits

                            # you can pass them to processor for postprocessing
                            b,c,h,w = controlnet_image.shape
                            seg_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h,w)]*b) ## B H W
                            seg_maps = torch.stack(seg_maps).cpu().numpy()
                        
                        tmp_labels = label_transform(batch['labels'], args.task_name, args.dataset_name, output_size=controlnet_image.shape[-2:])
                        for seg_map,label,condition in zip(seg_maps,tmp_labels,controlnet_image):
                            if 'ADE' in args.dataset_name:
                                seg_map[label.cpu() == 255] = 0                        
                            colored_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8) # height, width, 3

                            for label, color in enumerate(palette):
                                colored_seg[seg_map == label, :] = color
                            mask = (condition == 0).all(0).cpu().numpy()
                            colored_seg[mask == 1,:] = 0
                            losers.append(torch.tensor(colored_seg).permute(2,0,1)/255)
                        losers = torch.stack(losers).to(device = accelerator.device,dtype = weight_dtype)
                '''
                tmp = torchvision.transforms.functional.to_pil_image(losers[0].cpu())
                if args.task_name in ['hed','canny','lineart','depth']:
                    tmp = tmp.convert("L")
                tmp.save('loser.png')
                tmp = torchvision.transforms.functional.to_pil_image(controlnet_image[0].cpu())
                if args.task_name in ['hed','canny','lineart','depth']:
                    tmp = tmp.convert("L")
                tmp.save('winner.png')
                '''
                """
                Training ControlNet
                """
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(args.timestep_sampling_start, args.timestep_sampling_end, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                pretrain_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")


                """
                CPO
                """

                model_controlnet_image = torch.cat([controlnet_image.to(dtype=weight_dtype).clone(),losers.to(dtype=weight_dtype).clone()])  # image condition

                model_latents = latents.repeat(2,1,1,1) ## latents have been multiplied with scaling factor

                model_noise = torch.randn_like(latents).repeat(2,1,1,1) ## repeat the same noise for them, use different noise for contrastive
                model_timesteps = torch.randint(args.timestep_sampling_start, args.timestep_sampling_end, (bsz,), device=model_latents.device)
                model_timesteps = model_timesteps.long()
                dpo_mask = (model_timesteps > 0).float()
                model_timesteps = model_timesteps.repeat(2)

                model_encoder_hidden_states = encoder_hidden_states.repeat(2,1,1)
                model_noisy_latents = noise_scheduler.add_noise(model_latents, model_noise, model_timesteps)

                model_down_block_res_samples, model_mid_block_res_sample = controlnet(
                    model_noisy_latents,
                    model_timesteps,
                    encoder_hidden_states=model_encoder_hidden_states,
                    controlnet_cond=model_controlnet_image,
                    return_dict=False,
                )
                
                model_pred = unet(
                    model_noisy_latents,
                    model_timesteps,
                    encoder_hidden_states=model_encoder_hidden_states,
                    down_block_additional_residuals=[
                        model_sample.to(dtype=weight_dtype) for model_sample in model_down_block_res_samples
                    ],
                    mid_block_additional_residual=model_mid_block_res_sample.to(dtype=weight_dtype),
                ).sample
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = model_noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_latents, model_noise, model_timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                model_losses = (model_pred.float() - target.float()).pow(2).mean(dim=[1,2,3],keepdim=False)
                model_losses_w, model_losses_l = model_losses.chunk(2)
                

                
            
                with torch.no_grad(): # Get the reference policy (unet) prediction

                    ref_controlnet_image = torch.cat([controlnet_image.to(dtype=weight_dtype).clone(),losers.to(dtype=weight_dtype).clone()])  # image condition
                    
                    ref_latents =  latents.repeat(2,1,1,1)
                    ref_noise = model_noise.clone() ## repeat the same noise for them
                    ref_timesteps = model_timesteps.clone()
                    ref_encoder_hidden_states = encoder_hidden_states.repeat(2,1,1)
                    ref_noisy_latents = noise_scheduler.add_noise(ref_latents, ref_noise, ref_timesteps)

                    ref_down_block_res_samples, ref_mid_block_res_sample = ref_controlnet(
                        ref_noisy_latents,
                        ref_timesteps,
                        encoder_hidden_states=ref_encoder_hidden_states,
                        controlnet_cond=ref_controlnet_image,
                        return_dict=False,
                    )
                    
                    ref_pred = unet(
                        ref_noisy_latents,
                        ref_timesteps,
                        encoder_hidden_states=ref_encoder_hidden_states,
                        down_block_additional_residuals=[
                            ref_sample.to(dtype=weight_dtype) for ref_sample in ref_down_block_res_samples
                        ],
                        mid_block_additional_residual=ref_mid_block_res_sample.to(dtype=weight_dtype),
                    ).sample
                    
                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        ref_target = ref_noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        ref_target = noise_scheduler.get_velocity(ref_latents, ref_noise, ref_timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    
                    ref_losses = (ref_pred.float() - ref_target.float()).pow(2).mean(dim=[1,2,3],keepdim=False)                  

                    ref_losses_w, ref_losses_l = ref_losses.chunk(2)



                model_diff = (model_losses_w - model_losses_l).clone().detach() # These are both LBS (as is t)
                ref_diff = ref_losses_w - ref_losses_l # These are both LBS (as is t)
                scale_term = -0.5 * args.beta_dpo
                scale_term = 1/((scale_term * (model_diff - ref_diff)).detach().exp() + 1)
                #DPO_loss  = scale_term * (torch.clamp(model_diff + args.margin,min=0.002))
                
                DPO_loss =  torch.clamp(model_losses_w - model_losses_l + args.margin,min=0.)
                #DPO_loss =  model_losses_w - model_losses_l
                reward_loss = DPO_loss.clone().detach().mean()

                DPO_loss = scale_term * DPO_loss
                #DPO_loss =  torch.clamp(model_losses_w - model_losses_l + args.margin,min=0.)
                if dpo_mask.sum() == 0:
                    DPO_loss = DPO_loss.mean() * 0
                else:
                    DPO_loss = (DPO_loss * dpo_mask).sum()/dpo_mask.sum()
                """
                Losses
                """
                # Gather the losses across all processes for logging (if we use distributed training).
                
                loss = (args.grad_scale * pretrain_loss + DPO_loss).detach()
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_pretrain_loss = accelerator.gather(pretrain_loss.repeat(args.train_batch_size)).mean()
                avg_reward_loss = accelerator.gather(reward_loss.repeat(args.train_batch_size)).mean()

                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_pretrain_loss += avg_pretrain_loss.item() / args.gradient_accumulation_steps
                train_reward_loss += avg_reward_loss.item() / args.gradient_accumulation_steps

                # Back propagate
                #accelerator.backward(loss)
                accelerator.backward(args.grad_scale * pretrain_loss,retain_graph = True)
                accelerator.backward(DPO_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_controlnet.step(controlnet.parameters())
                progress_bar.update(1)
                global_step += 1

                # loss when perform gradient backward
                accelerator.log({
                        "train_loss": train_loss,
                        "train_pretrain_loss": train_pretrain_loss,
                        "train_reward_loss": train_reward_loss,
                        "lr": lr_scheduler.get_last_lr()[0]
                    },
                    step=global_step
                )
                loss_per_epoch += train_loss
                pretrain_loss_per_epoch += train_pretrain_loss
                reward_loss_per_epoch += train_reward_loss

                train_loss, train_pretrain_loss, train_reward_loss = 0., 0., 0.

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # directly save the state_dict
                        if accelerator.distributed_type != accelerate.DistributedType.FSDP:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)

                            if args.use_ema:
                                ema_controlnet.save_pretrained(f'{save_path}/controlnet_ema')

                            logger.info(f"Saved state to {save_path}")

                    start_time = time.time()
                    if global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            ema_controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            val_dataset
                        )

                        end_time = time.time()
                        logger.info(f"Validation time: {end_time - start_time} seconds")

            # only show in the progress bar
            logs = {
                "loss_step": loss.detach().item(),
                "pretrain_loss_step": pretrain_loss.detach().item(),
                "reward_loss_step": reward_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)

            # FSDP save model need to call all the ranks
            if global_step % args.checkpointing_steps == 0:
                if accelerator.distributed_type == accelerate.DistributedType.FSDP:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved accelerator state to {save_path}")

                    # Gather all of the state in the rank 0 device
                    accelerator.wait_for_everyone()
                    with FSDP.state_dict_type(controlnet, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                        state_dict = accelerator.get_state_dict(controlnet)

                    # Saving FSDP state
                    if accelerator.is_main_process:
                        torch.save(state_dict, os.path.join(save_path, 'controlnet_state_dict.pt'))
                        logger.info(f"Saved ControlNet state to {save_path}")

            if global_step >= args.max_train_steps:
                break

        logs = {
            "loss_epoch": loss_per_epoch * args.gradient_accumulation_steps / len(train_dataloader),
            "pretrain_loss_epoch": pretrain_loss_per_epoch * args.gradient_accumulation_steps / len(train_dataloader),
            "reward_loss_epoch": reward_loss_per_epoch * args.gradient_accumulation_steps / len(train_dataloader),
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()

    # If we use FSDP, saving the state_dict
    if accelerator.distributed_type == accelerate.DistributedType.FSDP:
        with FSDP.state_dict_type(controlnet, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            state_dict = accelerator.get_state_dict(controlnet)
            ema_state_dict = accelerator.get_state_dict(ema_controlnet)

        if accelerator.is_main_process:
            torch.save(state_dict, os.path.join(args.output_dir, 'controlnet_state_dict.pt'))
            torch.save(ema_state_dict, os.path.join(args.output_dir, 'controlnet_state_dict_ema.pt'))
            logger.info(f"Saved ControlNet state to {args.output_dir}")
    else:
        controlnet = accelerator.unwrap_model(controlnet)

        controlnet.save_pretrained(args.output_dir)
        ema_controlnet.save_pretrained(args.output_dir + '_ema')

    if accelerator.is_main_process:
        if args.push_to_hub:
            for _ in range(100):
                try:
                    save_model_card(
                        repo_id,
                        image_logs=image_logs,
                        base_model=args.pretrained_model_name_or_path,
                        repo_folder=args.output_dir,
                    )
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        path_in_repo=args.output_dir.replace('work_dirs/', ''),
                        commit_message=f"End of training {args.output_dir.split('/')[-1]}",
                        ignore_patterns=["step_*", "epoch_*"],
                        token=args.hub_token
                    )
                    break
                except:
                    continue

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)