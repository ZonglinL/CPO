import os
import time
import random
import torch
import numpy as np
import torchvision.transforms.functional as F
import argparse
import cv2
import torchvision
from scipy.optimize import linear_sum_assignment

from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from diffusers import (
    T2IAdapter, StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline, ControlNetModel,
    UniPCMultistepScheduler, DDIMScheduler,
    StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetPipeline, AutoencoderKL
)
from utils import get_reward_model, get_reward_loss ,label_transform 
from datasets import load_dataset, load_from_disk
from accelerate import PartialState
from PIL import Image 
from kornia.filters import canny
from transformers import DPTImageProcessor, DPTForDepthEstimation

from torchvision.transforms import Compose, Normalize, ToTensor
transforms = Compose([
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

from mmpose.codecs import SPR,AssociativeEmbedding
from PIL import PngImagePlugin,Image
MaximumDecompressedsize = 1024 
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedsize * MegaByte
Image.MAX_IMAGE_PIXELS = None
import mmcv
from itertools import product
import math

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


import sys

from PIL import Image, ImageDraw, ImageFont
import clip
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from torchvision.ops import nms,box_iou
from ultralytics import YOLO




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

    # Step 3: Normalize
    img_h, img_w = result.orig_shape

    bboxes_xywh[:, [0, 2]] /= img_w
    bboxes_xywh[:, [1, 3]] /= img_h

    keypoints[..., 0] /= img_w
    keypoints[..., 1] /= img_h

    return bboxes_xywh, keypoints, visibility




def oks_overlaps(kpt_preds, kpt_gts, kpt_valids, kpt_areas, sigmas, kpt_pred_vis):
    # Convert sigmas to a tensor and compute variances.
    sigmas = kpt_preds.new_tensor(sigmas)
    variances = (sigmas * 2) ** 2

    # Check that the number of predictions matches the number of ground truths.
    if kpt_preds.size(0) != kpt_gts.size(0):
        raise ValueError("Mismatch in number of predictions and ground truths.")
    
    # Add a batch dimension.
    kpt_preds = kpt_preds.unsqueeze(0)
    kpt_gts = kpt_gts.unsqueeze(0)
    kpt_areas = kpt_areas.unsqueeze(0)
    kpt_valids = kpt_valids.unsqueeze(0)
    kpt_pred_vis = kpt_pred_vis.unsqueeze(0)

    # Compute the squared Euclidean distance between predicted and ground-truth keypoints.
    squared_distance = (kpt_preds[:, :, 0] - kpt_gts[:, :, 0]) ** 2 + \
                       (kpt_preds[:, :, 1] - kpt_gts[:, :, 1]) ** 2

    # Normalize the squared distances.
    squared_distance_norm = squared_distance / (kpt_areas[:, None] * variances[None, :] * 2)
    exp_term = torch.exp(-squared_distance_norm)
    
    # Create a mask that is 1 when ground-truth visibility equals predicted visibility and 0 otherwise.
    vis_mask = (kpt_valids == kpt_pred_vis).float()
    
    # For keypoints where both the ground-truth and predicted keypoints are not visible,
    # force the similarity score to 1.
    both_invis = ((kpt_valids == 0) & (kpt_pred_vis == 0))
    exp_term[both_invis] = 1
    
    # Only keypoints with matching visibility contribute.
    exp_term = exp_term * vis_mask

    # The denominator is the count of keypoints where visibilities agree.
    denominator = vis_mask.sum(dim=1)
    
    # Compute the OKS, adding a small epsilon to avoid division by zero.
    oks = exp_term.sum(dim=1) / (denominator + 1e-6)

    return oks



def compute_oks(gt_keypoints, pred_keypoints, scales, k=0.5):
    """
    Compute OKS (Object Keypoint Similarity) for all GT-Pred pairs, skipping non-visible GT keypoints.

    Parameters:
        gt_keypoints: [N, 17, 3] numpy array (x, y, v) for N ground truth poses.
        pred_keypoints: [M, 17, 3] numpy array (x, y, v) for M predicted poses.
        scales: [N] numpy array of object scales for each GT.
        k: float, empirical keypoint constant.

    Returns:
        oks_matrix: [N, M] OKS similarity scores.
    """
    sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ], dtype=np.float32) / 10.0
    N, M = len(gt_keypoints), len(pred_keypoints)
    oks_matrix = torch.zeros((N, M)).to(gt_keypoints.device)

    for i in range(N):  # Loop over GT poses
        g = gt_keypoints[i]
        xg, yg, vg = g[:, 0], g[:, 1], g[:, 2]
        visible_mask = vg > 0  # GT visibility mask
        
        if visible_mask.sum() == 0:
            continue  # Skip if no GT keypoints are visible
        
        for j in range(M):  # Loop over predicted poses
            oks = oks_overlaps(pred_keypoints[j,:,:2],gt_keypoints[i,:,:2],gt_keypoints[i,:,-1],scales[i],sigmas,pred_keypoints[j,:,-1])
            oks_matrix[i, j] = oks

    return oks_matrix


def compute_ap(gt_keypoints, pred_keypoints, scales, oks_thresholds=[0.5 + 0.05 * i for i in range(10)]):
    """
    Compute Average Precision (AP) at different OKS thresholds and mean Average Precision (mAP),
    ensuring each GT is matched.

    Parameters:
        gt_keypoints: [N, 17, 3] numpy array of GT keypoints.
        pred_keypoints: [M, 17, 3] numpy array of predicted keypoints.
        scales: [N] numpy array of object scales.
        oks_thresholds: List of OKS thresholds for AP computation.

    Returns:
        ap_scores: Dictionary mapping each threshold to its AP.
        mAP: Mean Average Precision.
    """

    oks_matrix = compute_oks(gt_keypoints, pred_keypoints, scales)
    ap_scores = {}

    if len(pred_keypoints) == 0:
        for oks_thresh in oks_thresholds:
            ap_scores[oks_thresh] = 0

    else:
        oks_matrix = oks_matrix.max(1,keepdims = True).values

        for oks_thresh in oks_thresholds:

            if oks_matrix.shape[0] * oks_matrix.shape[1] == 0:
                ap = 0
            else:
                ap = (oks_matrix >= oks_thresh).sum()/(oks_matrix.shape[0] * oks_matrix.shape[1])
            #ap = np.mean(precision_interp)
            ap_scores[oks_thresh] = ap.cpu().numpy()

    # Compute mean AP across all thresholds
    mAP = np.mean(list(ap_scores.values()))

    return ap_scores, mAP


def estimate_scales_from_keypoints(gt_keypoints):
    """
    Estimate object scales based on keypoint spread.
    
    Parameters:
        gt_keypoints: [N, 17, 3] numpy array (x, y, v) for N ground truth poses.
    
    Returns:
        scales: [N] numpy array of estimated scales.
    """
    # Get visible keypoints only (where v > 0)
    visible = gt_keypoints[:, :, 2] > 0
    x_coords = gt_keypoints[:, :, 0] * visible  # Mask out invisible keypoints
    y_coords = gt_keypoints[:, :, 1] * visible

    # Compute bounding region spread
    x_spread = np.max(x_coords, axis=1) - np.min(x_coords, axis=1)
    y_spread = np.max(y_coords, axis=1) - np.min(y_coords, axis=1)

    # Scale estimate
    scales = x_spread * y_spread
    return scales

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def main(args):
    distributed_state = PartialState()
    #seed_torch(args.seed)
    box_threshold = 0.1

    iou_threshold = 0.9

    # load_dataset
    if args.dataset_name.count('/') == 1:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            cache_dir=args.cache_dir,
            split=args.dataset_split
        )
    else:
         # Loading from local disk.
        dataset = load_from_disk(
            dataset_path=args.dataset_name,
            split=args.dataset_split
        )


    with distributed_state.main_process_first():
        if args.task_name == 'depth':
            processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        elif args.task_name == 'lineart':
            from utils import get_reward_model
            model = get_reward_model(task='lineart', model_path='https://huggingface.co/spaces/awacke1/Image-to-Line-Drawings/resolve/main/model.pth')
            model.eval()
        elif args.task_name == 'hed':
            from utils import get_reward_model
            model = get_reward_model(task='hed', model_path='https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth')
            model.eval()
        elif args.task_name == 'pose':
            #from utils import get_reward_model
            #model = get_reward_model(task='pose',model_path="mmpose::body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512.py")
            model = YOLO("yolo11x-pose.pt")
            model.eval()
    model = model.to(distributed_state.device)
    # only the main process will create the output directory
    save_dir = os.path.join(args.output_dir, args.dataset_name.split('/')[-1], args.dataset_split, args.model_path.replace('/', '_'))

    save_dir = save_dir + '_' + str(args.guidance_scale) + '-' + str(args.num_inference_steps)
    if distributed_state.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(os.path.join(save_dir, "images"))
            os.makedirs(os.path.join(save_dir, "annotations"))
            os.makedirs(os.path.join(save_dir, "visualization"))
            os.makedirs(os.path.join(save_dir, "real_image"))

        for i in range(args.batch_size):
            if not os.path.exists(os.path.join(save_dir, f"images/group_{i}")):
                os.makedirs(os.path.join(save_dir, f"images/group_{i}"))

    if distributed_state.is_main_process:
        start_time = time.time()

    # split dataset into multiple processes and gpus, each process corresponds to a gpu
    print(len(dataset))

    codec = SPR(
                input_size=(args.resolution, args.resolution),
                heatmap_size=(args.resolution//4, args.resolution//4),
                sigma=(4, 2),
                minimal_diagonal_length=32**0.5,
                generate_keypoint_heatmaps=True,
                decode_max_instances=30
            )
    from mmengine.hub import get_model  # segmentation

    model = get_model("mmpose::body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512.py", pretrained=True).to(distributed_state.device)
    total_loss = 0
    counter = 0
    with distributed_state.split_between_processes(list(range(len(dataset)))) as local_idxs:

        print(f"{distributed_state.process_index} has {len(local_idxs)} images")
        ap5 = []
        ap75 = []
        mean_ap = []

        for idx in local_idxs:
            # Unique Identifier, used for saving images while avoid overwriting due to the same prompt
            #print(f"Processing image {idx}...")
            uid = str(idx)

            original_image = dataset[idx][args.image_column].convert('RGB')
            
            resize_img = original_image.resize((args.resolution, args.resolution), Image.Resampling.BICUBIC)
            label = dataset[idx][args.label_column] if args.label_column is not None else None

            label = np.array(label)
            valid = np.where((label[:,:,2] != 0).sum(-1) != 0)## visible poses
            label = label[valid]
            
            pose = label[:, :, :2]            
            if (np.array(pose) == 0).all():
                continue

            poses_vis = label[:, :, 2]
            
            mask = (poses_vis != 0).any(-1)
            pose = pose[mask]
            poses_vis = poses_vis[mask]
            if len(pose) == 0:
                continue
            #print(f"Processing image {idx}...")
            input_image_size = np.array(original_image.size)             # width, height
            output_imgae_size = np.array([args.resolution, args.resolution])

            pose = pose*output_imgae_size/input_image_size
            gt_key = np.concatenate([pose,np.expand_dims(poses_vis, axis=-1)],axis = -1)
            

            images = [Image.open(f"{save_dir}/images/group_{i}/{uid}.png") for i in range(4)]
            transform = torchvision.transforms.ToTensor()
            inputs = [transform(images[i]) for i in range(len(images))]
    
            inputs = torch.stack(inputs).to(device = distributed_state.device)

            scales = estimate_scales_from_keypoints(gt_key) + 1e-8
            scales = torch.tensor(scales).to(distributed_state.device)
            gt_key = torch.tensor(gt_key).to(distributed_state.device)
            with torch.no_grad():
                for i in range(4):
                    
                    feats = model.extract_feat(inputs[i].unsqueeze(0).to(distributed_state.device))
                    outputs = model.head(feats)
                    outputs = torch.cat(outputs, dim=1)
                    keypoints_filt,(_,pred_vis) = codec.decode(outputs[0,:18],outputs[0,18:])
                    keypoints_filt = keypoints_filt.squeeze(0)
                    pred_vis = pred_vis.squeeze(0)
                    pred_vis = pred_vis > 0.5

                    """
                    results = model(images[i], verbose=False)[0]
                    boxes_filt,keypoints_filt,pred_vis = filter_yolo_pose_results(results)
                    pred_vis = (pred_vis > 0.5).float()
                    """
                    
                    mask = (pred_vis != 0).any(-1)
                    keypoints_filt = keypoints_filt[mask]
                    #keypoints_filt *= args.resolution
                    pred_vis = pred_vis[mask]
                    
                    pred_key = torch.cat([keypoints_filt,pred_vis.unsqueeze(-1)],-1)
                    

                    oks_mat = compute_oks(gt_key,pred_key,scales) + 1e-8
                    if len(pred_key) == 0:
                        tmp_loss = 2
                    else:
                        tmp_loss = (1 - oks_mat.max(0).values).mean() + (1-oks_mat.max(1).values).mean() # linear scale loss  
                    total_loss += tmp_loss  

                    ap_results, mAP = compute_ap(gt_key, pred_key, scales)
                    ap5.append(ap_results[0.5])
                    ap75.append(ap_results[0.75])
                    mean_ap.append(mAP)

                    counter += 1
        print(total_loss/counter)
        print(f"AP0.5:{np.mean(ap5)}")
        print(f"AP0.75:{np.mean(ap75)}")
        print(f"MAP:{np.mean(mean_ap)}")
    
        np.savetxt(os.path.join(f'{save_dir}','mAP.txt'), [np.mean(mean_ap)])
        

    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
        end_time = time.time()
        print(f"Validation time: {end_time - start_time} seconds")

if __name__ == "__main__":
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
    main(args)


""" ControlNet """
# Hed: accelerate launch --main_process_port=23333 --num_processes=4 controlnet/eval.py --task_name='hed' --dataset_name='limingcv/MultiGen-20M_canny_eval' --dataset_split='validation' --condition_column='image' --prompt_column='text' --model_path='lllyasviel/control_v11p_sd15_softedge'

# LineDrawing: accelerate launch --main_process_port=23333 --num_processes=4 controlnet/eval.py --task_name='lineart' --dataset_name='limingcv/MultiGen-20M_canny_eval' --dataset_split='validation' --condition_column='image' --prompt_column='text' --model_path='lllyasviel/control_v11p_sd15_lineart'

# COCOStuff: accelerate launch --main_process_port=23456 --num_processes=4 controlnet/eval.py --task_name='seg' --dataset_name='limingcv/Captioned_COCOStuff' --dataset_split='validation' --condition_column='control_seg' --prompt_column='prompt' --label_column='panoptic_seg_map' --model_path='lllyasviel/control_v11p_sd15_seg'

# ADE20K: accelerate launch --main_process_port=23456 --num_processes=4 controlnet/eval.py --task_name='seg' --dataset_name='limingcv/Captioned_ADE20K' --dataset_split='validation' --condition_column='control_seg' --prompt_column='prompt' --label_column='seg_map' --model_path='lllyasviel/control_v11p_sd15_seg'

# Canny: accelerate launch --main_process_port=12456 --num_processes=4 controlnet/eval.py --task_name='canny' --dataset_name='limingcv/MultiGen-20M_canny_eval' --dataset_split='validation' --condition_column='image' --prompt_column='text' --model_path='lllyasviel/control_v11p_sd15_canny'

# Depth: accelerate launch --main_process_port=12456 --num_processes=4 controlnet/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --prompt_column='text'  --label_column='control_depth' --model_path='lllyasviel/control_v11f1p_sd15_depth'


""" ControlNet-SDXL """
# Canny: accelerate launch --main_process_port=12456 --num_processes=4 controlnet/eval.py --task_name='canny' --dataset_name='limingcv/MultiGen-20M_canny_eval' --dataset_split='validation' --condition_column='image' --prompt_column='text' --model_path='diffusers/controlnet-canny-sdxl-1.0' --sd_path='stabilityai/stable-diffusion-xl-base-1.0' --model='controlnet-sdxl' --num_inference_steps=50 --resolution 1024

# Depth: accelerate launch --main_process_port=12456 --num_processes=4 controlnet/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --prompt_column='text' --label_column='control_depth'  --model_path='diffusers/controlnet-depth-sdxl-1.0' --sd_path='stabilityai/stable-diffusion-xl-base-1.0' --model='controlnet-sdxl' --num_inference_steps=50 --resolution 1024


""" T2I-Adapter """
# Depth: accelerate launch --main_process_port=12456 --num_processes=4 controlnet/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --prompt_column='text'  --label_column='control_depth' --model_path='TencentARC/t2iadapter_depth_sd15v2' --sd_path='stable-diffusion-v1-5/stable-diffusion-v1-5' --model='t2i-adapter' --num_inference_steps=50

# Canny: accelerate launch --main_process_port=23333 --num_processes=4 controlnet/eval.py --task_name='canny' --dataset_name='limingcv/MultiGen-20M_canny_eval' --dataset_split='validation' --condition_column='image' --prompt_column='text' --model_path='TencentARC/t2iadapter_canny_sd15v2' --sd_path='stable-diffusion-v1-5/stable-diffusion-v1-5' --model='t2i-adapter' --num_inference_steps=50

# ADE20K: accelerate launch --main_process_port=23333 --num_processes=4 controlnet/eval.py --task_name='seg' --dataset_name='limingcv/Captioned_ADE20K' --dataset_split='validation' --condition_column='control_seg' --prompt_column='prompt' --label_column='seg_map' --model_path='TencentARC/t2iadapter_seg_sd14v1' --sd_path='CompVis/stable-diffusion-v1-4' --model='t2i-adapter' --num_inference_steps=50


""" T2I-Adapter-SDXL """
# Canny: accelerate launch --main_process_port=23333 --num_processes=4 controlnet/eval.py --task_name='canny' --dataset_name='limingcv/MultiGen-20M_canny_eval' --dataset_split='validation' --condition_column='image' --prompt_column='text' --model_path='TencentARC/t2i-adapter-canny-sdxl-1.0' --sd_path='stabilityai/stable-diffusion-xl-base-1.0' --model='t2i-adapter-sdxl' --num_inference_steps=50 --resolution 1024

# Depth: accelerate launch --main_process_port=12456 --num_processes=4 controlnet/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --prompt_column='text'  --label_column='control_depth' --model_path='TencentARC/t2i-adapter-depth-midas-sdxl-1.0' --sd_path='stabilityai/stable-diffusion-xl-base-1.0' --model='t2i-adapter-sdxl' --num_inference_steps=50 --resolution 1024

# LineArt: accelerate launch --main_process_port=23333 --num_processes=4 controlnet/eval.py --task_name='lineart' --dataset_name='limingcv/MultiGen-20M_canny_eval' --dataset_split='validation' --condition_column='image' --prompt_column='text' --model_path='TencentARC/t2i-adapter-lineart-sdxl-1.0' --sd_path='stabilityai/stable-diffusion-xl-base-1.0' --model='t2i-adapter-sdxl' --num_inference_steps=50 --resolution 1024