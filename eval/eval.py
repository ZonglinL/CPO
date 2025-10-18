import os
import time
import random
import torch
import numpy as np
import torchvision.transforms.functional as F
import argparse
import cv2

from matplotlib import pyplot as plt
from torchvision.utils import make_grid, save_image
from diffusers import (
    T2IAdapter, StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler, DDIMScheduler,
    StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetPipeline, AutoencoderKL, DiffusionPipeline,FluxTransformer2DModel
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controlnets.controlnet_flux import FluxControlNetModel

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
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation,AutoModelForDepthEstimation
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from PIL import PngImagePlugin
MaximumDecompressedsize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedsize * MegaByte
Image.MAX_IMAGE_PIXELS = None

from PIL import Image
from huggingface_hub import snapshot_download
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline



import io, gc


def pil_to_heatmap(pil_img, colormap=cv2.COLORMAP_VIRIDIS):
    """
    Convert a grayscale PIL image to a heatmap using OpenCV.
    
    Args:
        pil_img (PIL.Image): Input grayscale image.
        colormap (int): OpenCV colormap to apply.

    Returns:
        PIL.Image: Heatmapped image.
    """
    # Ensure grayscale
    pil_img = pil_img.convert("L")
    
    # Convert to numpy array
    gray_np = np.array(pil_img)

    # Apply OpenCV colormap
    heatmap_np = cv2.applyColorMap(gray_np, colormap)

    # Convert back to PIL
    heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap_np, cv2.COLOR_BGR2RGB))
    return heatmap_pil

# Example usage:
# pil_img = Image.open("your_image.png").convert("L")
# heatmap_img = pil_to_heatmap(pil_img)
# heatmap_img.show() or heatmap_img.save("heatmap_output.png")


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False

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
    seed_torch(args.seed)

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

    print(f"Loading pre-trained weights from {args.model_path}")

    # main_process_first: Avoid repeated downloading of models for all processes
    with distributed_state.main_process_first():
        # load pre-trained model
        if args.model == 'controlnet':

            controlnet = ControlNetModel.from_pretrained(args.model_path, torch_dtype=torch.float16)

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                pretrained_model_name_or_path=args.sd_path,
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch.float16
            )
        elif args.model == 'controlnet-FLUX':
            controlnet = FluxControlNetModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
            transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

            pipe = FluxControlNetPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                controlnet=controlnet,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )

        elif args.model == 'controlnet-sdxl':
            controlnet = ControlNetModel.from_pretrained(args.model_path, torch_dtype=torch.float16)
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                args.sd_path,  # "stabilityai/stable-diffusion-xl-base-1.0"
                controlnet=controlnet,
                vae=vae,
                torch_dtype=torch.float16,
            )
        elif args.model == 't2i-adapter':
            adapter = T2IAdapter.from_pretrained(args.model_path, torch_dtype=torch.float16)
            pipe = StableDiffusionAdapterPipeline.from_pretrained(
                args.sd_path, adapter=adapter, safety_checker=None, torch_dtype=torch.float16, variant="fp16"
            )
        elif args.model == 't2i-adapter-sdxl':
            adapter = T2IAdapter.from_pretrained(args.model_path, torch_dtype=torch.float16, varient="fp16")
            euler_a = EulerAncestralDiscreteScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
            vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
                args.sd_path, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16",
            )
        else:
            raise NotImplementedError(f"Model {args.model} not implemented")

    pipe.to(distributed_state.device)

    with distributed_state.main_process_first():
        if args.task_name == 'depth':
            image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            #model = get_reward_model(task = 'depth').to(distributed_state.device)
            model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(distributed_state.device)
            model.eval()
        elif args.task_name == 'lineart':
            from utils import get_reward_model
            model = get_reward_model(task='lineart', model_path='https://huggingface.co/spaces/awacke1/Image-to-Line-Drawings/resolve/main/model.pth').to(distributed_state.device)
            model.eval()
        elif args.task_name == 'hed':
            from utils import get_reward_model
            model = get_reward_model(task='hed', model_path='https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth').to(distributed_state.device)
            model.eval()
        elif args.task_name == 'pose':
            from utils import get_reward_model
            model = get_reward_model(task='pose',model_path="mmpose::body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512.py").to(distributed_state.device)
            model.eval()
        elif args.task_name == "seg":
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
                image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-swin-large")
                model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-large")
                model.to(distributed_state.device)
                model.eval()

                
                
            else:
                classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
                palette = [[220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30], [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157], [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240], [0, 125, 92], [209, 0, 151], [188, 208, 182], [0, 220, 176], [255, 99, 164], [92, 0, 73], [133, 129, 255], [78, 180, 255], [0, 228, 0], [174, 255, 243], [45, 89, 255], [134, 134, 103], [145, 148, 174], [255, 208, 186], [197, 226, 255], [171, 134, 1], [109, 63, 54], [207, 138, 255], [151, 0, 95], [9, 80, 61], [84, 105, 51], [74, 65, 105], [166, 196, 102], [208, 195, 210], [255, 109, 65], [0, 143, 149], [179, 0, 194], [209, 99, 106], [5, 121, 0], [227, 255, 205], [147, 186, 208], [153, 69, 1], [3, 95, 161], [163, 255, 0], [119, 0, 170], [0, 182, 199], [0, 165, 120], [183, 130, 88], [95, 32, 0], [130, 114, 135], [110, 129, 133], [166, 74, 118], [219, 142, 185], [79, 210, 114], [178, 90, 62], [65, 70, 15], [127, 167, 115], [59, 105, 106], [142, 108, 45], [196, 172, 0], [95, 54, 80], [128, 76, 255], [201, 57, 1], [246, 0, 122], [191, 162, 208], [255, 255, 128], [147, 211, 203], [150, 100, 100], [168, 171, 172], [146, 112, 198], [210, 170, 100], [92, 136, 89], [218, 88, 184], [241, 129, 0], [217, 17, 255], [124, 74, 181], [70, 70, 70], [255, 228, 255], [154, 208, 0], [193, 0, 92], [76, 91, 113], [255, 180, 195], [106, 154, 176], [230, 150, 140], [60, 143, 255], [128, 64, 128], [92, 82, 55], [254, 212, 124], [73, 77, 174], [255, 160, 98], [255, 255, 255], [104, 84, 109], [169, 164, 131], [225, 199, 255], [137, 54, 74], [135, 158, 223], [7, 246, 231], [107, 255, 200], [58, 41, 149], [183, 121, 142], [255, 73, 97], [107, 142, 35], [190, 153, 153], [146, 139, 141], [70, 130, 180], [134, 199, 156], [209, 226, 140], [96, 36, 108], [96, 96, 96], [64, 170, 64], [152, 251, 152], [208, 229, 228], [206, 186, 171], [152, 161, 64], [116, 112, 0], [0, 114, 143], [102, 102, 156], [250, 141, 255]]

                image_processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-coco")
                model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")

                model.to(distributed_state.device)
                model.eval()
                
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

    if args.model == 'controlnet':
        if args.ddim:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # NOTE: assign a specific gpu_id is necessary, otherwise all models will be loaded on gpu 0
    pipe.enable_model_cpu_offload(gpu_id=distributed_state.process_index)

    if distributed_state.is_main_process:
        start_time = time.time()

    # split dataset into multiple processes and gpus, each process corresponds to a gpu
    print(len(dataset))
    with distributed_state.split_between_processes(list(range(len(dataset)))) as local_idxs:

        print(f"{distributed_state.process_index} has {len(local_idxs)} images")

        for idx in local_idxs:
            # Unique Identifier, used for saving images while avoid overwriting due to the same prompt
            print(f"Processing image {idx}...")
            uid = str(idx)

            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

            if os.path.exists(f'{save_dir}/visualization/{uid}.png'):
                continue

            original_image = dataset[idx][args.image_column].convert('RGB')#.resize((args.resolution, args.resolution), Image.Resampling.BICUBIC)
            image = original_image.resize((args.resolution, args.resolution), Image.Resampling.BICUBIC)
            condition = dataset[idx][args.condition_column].convert('RGB').resize((args.resolution, args.resolution), Image.Resampling.BICUBIC)
            prompt = dataset[idx][args.prompt_column]
            label = dataset[idx][args.label_column] if args.label_column is not None else None

            if args.task_name == 'canny':
                low_threshold = 0.1 # random.uniform(0, 1)
                high_threshold = 0.2 # random.uniform(low_threshold, 1)
                condition = canny(F.pil_to_tensor(condition).unsqueeze(0) / 255.0, low_threshold, high_threshold)[1]
                condition = F.to_pil_image(condition.squeeze(0), 'L')
                condition = condition.convert('RGB') if args.model != 't2i-adapter' else condition
                label = condition
            if args.task_name == 'canny_cv2':
                low_threshold = 100
                high_threshold = 200

                condition = np.array(condition)[:, :, ::-1].copy()  # RGB to BGR
                condition = cv2.Canny(np.array(condition), low_threshold, high_threshold)
                condition = Image.fromarray(condition)
                condition = condition.convert('RGB') if args.model != 't2i-adapter' else condition
                label = condition

            elif args.task_name in ['lineart', 'hed']:
                condition = F.pil_to_tensor(condition.resize((args.resolution, args.resolution))).unsqueeze(0) / 255.0
                with torch.no_grad():
                    condition = model(condition.to(distributed_state.device)).cpu()

                condition = 1 - condition if args.task_name == 'lineart' else condition
                condition = condition.reshape(args.resolution, args.resolution)
                condition = F.to_pil_image(condition, 'L').convert('RGB')
                label = condition
            elif args.task_name == 'pose':

                label = np.array(label)
                pose = label[:,:,:2]
                if (pose == 0).all():
                    continue
                poses_vis = label[:, :, 2]
            
                mask = (poses_vis != 0).any(-1)
                pose = pose[mask]
                poses_vis = poses_vis[mask]
                if len(pose) == 0:
                    continue
                
                label = condition
            elif args.task_name == "depth":
                pass

            elif args.task_name == 'seg':
                pass      
            condition = condition.resize((args.resolution, args.resolution), Image.Resampling.NEAREST)
            prompts, conditions = [prompt] * args.batch_size, [condition] * args.batch_size
            


            if args.model == 't2i-adapter-sdxl' and args.task_name == 'lineart':
                images = pipe(
                    prompt=prompts,
                    image=conditions,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    adapter_conditioning_scale=0.5,
                ).images
            elif args.model == 'controlnet-FLUX':

                images = pipe(
                    prompts, 
                    control_image=conditions,
                    controlnet_conditioning_scale=0.7,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                ).images
            else:
                images = pipe(
                    prompt=prompts,
                    image=conditions,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    #generator = generator,

                    #negative_prompt=['worst quality, low quality'] *  args.batch_size
                ).images

            if args.task_name == 'canny':
                canny_edges = [F.pil_to_tensor(img)/255.0 for img in images]
                with torch.no_grad():
                    canny_edges = canny(torch.stack(canny_edges), low_threshold, high_threshold)[1]
                canny_edges = torch.chunk(canny_edges, args.batch_size, dim=0)
                canny_edges = [x.reshape(1, args.resolution, args.resolution) for x in canny_edges]
                canny_edges = [F.to_pil_image(x, 'L').convert('RGB') for x in canny_edges]
                [img.save(f"{save_dir}/images/group_{i}/{uid}_edge.png") for i, img in enumerate(canny_edges)]

            elif args.task_name == 'canny_cv2':
                canny_edges = [cv2.Canny(np.array(images[0])[:, :, ::-1].copy(), 100, 200) for img in images]
                canny_edges = [torch.tensor(x)/255.0 for x in canny_edges]
                canny_edges = [x.reshape(1, args.resolution, args.resolution) for x in canny_edges]
                canny_edges = [F.to_pil_image(x, 'L').convert('RGB') for x in canny_edges]
                [img.save(f"{save_dir}/images/group_{i}/{uid}_edge.png") for i, img in enumerate(canny_edges)]
            elif args.task_name in ['lineart', 'hed']:
                lineart = [F.pil_to_tensor(img)/255.0 for img in images]
                with torch.no_grad():
                    lineart = model(torch.stack(lineart).to(distributed_state.device)).cpu()
                lineart = torch.chunk(lineart, args.batch_size, dim=0)
                lineart = [x.reshape(1, args.resolution, args.resolution) for x in lineart]
                lineart = [F.to_pil_image(x).convert('RGB') for x in lineart]
                [img.save(f"{save_dir}/images/group_{i}/{uid}_lineart.png") for i, img in enumerate(lineart)]

            elif args.task_name == "depth":
                label = np.array(label)
                label = label - label.min()
                label = label / label.max()
                label = label * 255             
                #image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
                depth_model_input = image_processor(images=images, return_tensors="pt")
                depth_model_input = {k: v.to(distributed_state.device) for k, v in depth_model_input.items()}
                
                with torch.no_grad():
                    outputs = model(**depth_model_input) 
                    predicted_depth = outputs.predicted_depth.cpu() ## b h w
                    num_val = predicted_depth.shape[0]
                    min_values = predicted_depth.view(num_val, -1).amin(dim=1, keepdim=True).view(num_val, 1, 1)
                    predicted_depth = predicted_depth - min_values
                    max_values = predicted_depth.view(num_val, -1).amax(dim=1, keepdim=True).view(num_val, 1, 1)
                    predicted_depth = predicted_depth / max_values ## 0,1
                    #depth_maps = [F.to_pil_image(x/x.max()).convert('RGB').resize((args.resolution, args.resolution), Image.Resampling.BILINEAR) for x in predicted_depth]
                    depth_maps = [F.to_pil_image((x - x.min())/(x.max() - x.min())).convert('RGB').resize((args.resolution, args.resolution), Image.Resampling.BILINEAR) for x in predicted_depth]
                    [img.convert('L').save(f"{save_dir}/images/group_{i}/{uid}_depth.png") for i, img in enumerate(depth_maps)]
            elif args.task_name == "seg":
                inputs = image_processor(images=images, return_tensors="pt")
                inputs['pixel_values'] = inputs['pixel_values'].to(distributed_state.device)

                #print(inputs)
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                if "ADE" in args.dataset_name:
                    logits = outputs.logits  # shape (batch_size, num_labels, height, width)
                    seg_map = logits.argmax(1).cpu().numpy() + 1 ## B H W
                else:
                    target_sizes = [img.size[::-1] for img in images]
                    seg_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
                    seg_map = torch.stack(seg_map)
                    seg_map = seg_map.cpu().numpy() # BHW
                    
                
                seg_maps = []
                
                for j in range(4):

                    colored_seg = np.zeros((512, 512, 3), dtype=np.uint8) # height, width, 3

                    for l, color in enumerate(palette):
                        colored_seg[seg_map[j] == l, :] = color
 
                    colored_seg = Image.fromarray(colored_seg.astype(np.uint8))
                    
                    seg_maps.append(colored_seg)
            

            # save ground truth labels
            if label is not None:
                label = Image.fromarray(np.array(label).astype('uint8'))
                label.resize((args.resolution, args.resolution), Image.Resampling.NEAREST).save(f"{save_dir}/annotations/{uid}.png")

            # scale the generated images to the original resolution for evaluation
            # then save the generated images for evaluation
            [img.save(f"{save_dir}/images/group_{i}/{uid}.png") for i, img in enumerate(images)]
            original_image.save(f"{save_dir}/real_image/{uid}.png")

            # generate a grid of images
            if args.task_name in ['canny', 'canny_cv2']:
                # input image, condition image, generated_images
                images = [image] + images + [condition] + canny_edges
                images = [img.convert('RGB') for img in images] if args.model == 't2i-adapter' else images
                images = [F.pil_to_tensor(x) for x in images]
                images = make_grid(images, nrow=len(images)//2)
            elif args.task_name in ['lineart', 'hed']:
                # input image, condition image, generated_images
                if args.task_name == 'lineart':
                    condition = 255 - F.pil_to_tensor(condition)
                    condition = F.to_pil_image(condition)

                images = [image] + images + [condition] + lineart
                images = [img.convert('RGB') for img in images] if args.model == 't2i-adapter' else images
                images = [F.pil_to_tensor(x) for x in images]
                images = make_grid(images, nrow=len(images)//2)
            elif args.task_name == 'depth':
                # input image, condition image, generated_images
                images = [image] + images + [pil_to_heatmap(condition)] + [pil_to_heatmap(d) for d in depth_maps]
                images = [F.pil_to_tensor(x) for x in images]
                images = make_grid(images, nrow=len(images)//2)
            elif args.task_name == 'seg':
                images = [image] + images + [condition] + seg_maps
                images = [F.pil_to_tensor(x) for x in images]
                images = make_grid(images, nrow=len(images)//2)
            else:
                # input image, condition image, generated_images
                images = [image] + [condition] + images
                images = [F.pil_to_tensor(x) for x in images]
                images = make_grid(images, nrow=len(images))

            images =  images.float()/ 255.
            save_image(images, f'{save_dir}/visualization/{uid}.png')

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
    parser.add_argument('--seed', type=int, default=12345, help='Random seed for reproducibility')
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