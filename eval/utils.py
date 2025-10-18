import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as F

from PIL import Image
from typing import Optional
from functools import partial
from torch import Tensor
from torchvision import transforms

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from canny_tools import Canny  # canny edge detection
from mmengine.hub import get_model  # segmentation
from transformers import DPTForDepthEstimation,AutoModelForDepthEstimation # depth estimation

from mmseg.models.losses.silog_loss import silog_loss
from torchvision.transforms import RandomCrop
from mmpose.models.losses import KeypointMSELoss, SoftWeightSmoothL1Loss
from ultralytics import YOLO


def get_reward_model(task='segmentation', model_path='mmseg::upernet/upernet_r50_4xb4-160k_ade20k-512x512.py'):
    """Return reward model for different tasks.

    Args:
        task (str, optional): Task name. Defaults to 'segmentation'.
        model_path (str, optional): Model name or pre-trained path.

    """
    if task == 'segmentation':
        return get_model(model_path, pretrained=True)
    elif task == 'pose':
        return YOLO("yolo11x-pose.pt")
    elif task == 'canny':
        return Canny()
    elif task == 'depth':
        #return DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        folder = os.path.expanduser("~/.cache/share/depth_anything")
        try:
            model = AutoModelForDepthEstimation.from_pretrained(folder)
        except:
            model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
            model.save_pretrained(folder)
        return model
        #return AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    elif task == 'lineart':
        model = LineDrawingModel()
        try:
            state_dict = torch.load(os.path.expanduser("~/.cache/share/lineart.pth"), map_location=torch.device('cpu'))
        except:
            state_dict = torch.hub.load_state_dict_from_url("https://huggingface.co/spaces/awacke1/Image-to-Line-Drawings/resolve/main/model.pth", map_location=torch.device('cpu'))
            cached_path = hf_hub_download(
                repo_id="awacke1/Image-to-Line-Drawings",
                filename="model.pth",
                repo_type="space"
            )
            target_path = os.path.expanduser("~/.cache/share/lineart.pth")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(cached_path, target_path)
        
        model.load_state_dict(state_dict)
        return model
    elif task == 'hed':
        model = HEDdetector()
        
        try:
            state_dict = torch.load(os.path.expanduser("~/.cache/share/hed.pth"), map_location=torch.device('cpu'))
        except:
            state_dict = torch.hub.load_state_dict_from_url("https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth", map_location=torch.device('cpu'))
            cached_path = hf_hub_download(
                repo_id="lllyasviel/Annotators",
                filename="ControlNetHED.pth",
                repo_type="model"
            )
            target_path = os.path.expanduser("~/.cache/share/hed.pth")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(cached_path, target_path)

        model.netNetwork.load_state_dict(state_dict)
        return model
        
        #return HEDdetector("https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth")
    else:
        raise not NotImplementedError("Only support segmentation, canny and depth for now.")


def get_reward_loss(predictions, labels, task='segmentation', **args):
    """Return reward loss for different tasks.

    Args:
        task (str, optional): Task name.

    Returns:
        torch.nn.Module: Loss class.
    """
    if task == 'segmentation':
        return nn.functional.cross_entropy(predictions, labels, ignore_index=255,reduction = 'none', **args).mean((-1,-2))
    elif task == 'canny':
        loss = nn.functional.mse_loss(predictions, labels,reduction = 'none', **args)
        return loss.mean((-1,-2,-3))
    elif task in ['depth', 'lineart', 'hed']:
        loss = nn.functional.mse_loss(predictions, labels,reduction = 'none', **args)
        return loss.mean((-1,-2,-3))
    elif task == 'pose':
        num_keypoints = 17  # openpose has 17 keypoints

        keypoint_loss_fn = KeypointLoss(
            use_target_weight=True,
            loss_weight=1,
        )
        displacement_loss_fn = DisplacementLoss(
            use_target_weight=True,
            supervise_empty=False,
            beta=1 / 9,
            loss_weight=0.002,
        )

        heatmaps_pred, displacements_pred = predictions[:, :num_keypoints+1], predictions[:, num_keypoints+1:]
        heatmaps_info, displacements_info = labels[:, :(num_keypoints+1)*2], labels[:, (num_keypoints+1)*2:]

        heatmaps_gt, heatmap_weights = heatmaps_info.chunk(2, dim=1)
        displacements_gt, displacement_weights = displacements_info.chunk(2, dim=1)

        keypoint_loss = keypoint_loss_fn(heatmaps_pred, heatmaps_gt, heatmap_weights)
        displacement_loss = displacement_loss_fn(displacements_pred, displacements_gt, displacement_weights)

        return keypoint_loss + displacement_loss
    else:
        raise not NotImplementedError("Only support segmentation, canny, depth and pose for now.")


def image_grid(imgs, rows, cols):
    """Image grid for visualization."""
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def map_color_to_index(image, dataset='limingcv/Captioned_ADE20K'):
    """Map colored segmentation image (RGB) into original label format (L).

    Args:
        image (torch.tensor): image tensor with shape (N, 3, H, W).
        dataset (str, optional): Dataset name. Defaults to 'ADE20K'.

    Returns:
        torch.tensor: mask tensor with shape (N, H, W).
    """
    if dataset == 'limingcv/Captioned_ADE20K':
        palette = np.load('ade20k_palette.npy')
    elif dataset == 'limingcv/Captioned_COCOStuff':
        palette = np.load('coco_stuff_palette.npy')
    else:
        raise NotImplementedError("Only support ADE20K and COCO-Stuff dataset for now.")

    image = image * 255
    palette_tensor = torch.tensor(palette, dtype=image.dtype, device=image.device)
    reshaped_image = image.permute(0, 2, 3, 1).reshape(-1, 3)

    # Calculate the difference of colors and find the index of the minimum distance
    indices = torch.argmin(torch.norm(reshaped_image[:, None, :] - palette_tensor, dim=-1), dim=-1)

    # Transform indices back to original shape
    return indices.view(image.shape[0], image.shape[2], image.shape[3])


def seg_label_transform(
        labels,
        dataset_name='limingcv/Captioned_ADE20K',
        output_size=(64, 64),
        interpolation=transforms.InterpolationMode.NEAREST,
        max_size=None,
        antialias=True):
    """Adapt RGB seg_map into loss computation. \
    (1) Map the RGB seg_map into the original label format (Single Channel). \
    (2) Resize the seg_map into the same size as the output feature map. \
    (3) Remove background class if needed (usually for ADE20K).

    Args:
        labels (torch.tensor): Segmentation map. (N, 3, H, W) for ADE20K and (N, H, W) for COCO-Stuff.
        dataset_name (string): Dataset name. Default to 'ADE20K'.
        output_size (tuple): Resized image size, should be aligned with the output of segmentation models.
        interpolation (optional): _description_. Defaults to transforms.InterpolationMode.NEAREST.
        max_size (optional): Defaults to None.
        antialias (optional): Defaults to True.

    Returns:
        torch.tensor: formatted labels for loss computation.
    """

    if dataset_name == 'limingcv/Captioned_ADE20K':
        labels = map_color_to_index(labels, dataset_name)
        labels = F.resize(labels, output_size, interpolation, max_size, antialias)

        # 0 means the background class in ADE20K
        # In a unified format, we use 255 to represent the background class for both ADE20K and COCO-Stuff
        labels = labels - 1
        labels[labels == -1] = 255
    elif dataset_name == 'limingcv/Captioned_COCOStuff':
        labels = F.resize(labels, output_size, interpolation, max_size, antialias)

    return labels.long()

def pose_label_transform(
        labels,
        dataset_name,
        output_size=(64, 64),
        interpolation=transforms.InterpolationMode.NEAREST,
        max_size=None,
        antialias=True
    ):

    # labels = F.resize(labels, output_size, interpolation, max_size, antialias)
    return labels


def depth_label_transform(
        labels,
        dataset_name,
        output_size=None,
        interpolation=transforms.InterpolationMode.BILINEAR,
        max_size=None,
        antialias=True
    ):

    if output_size is not None:
        labels = F.resize(labels, output_size, interpolation, max_size, antialias)
    return labels


def edge_label_transform(labels, dataset_name):
    return labels


def label_transform(labels, task, dataset_name, **args):
    if task == 'segmentation':
        return seg_label_transform(labels, dataset_name, **args)
    elif task == 'depth':
        return depth_label_transform(labels, dataset_name, **args)
    elif task == 'pose':
        return pose_label_transform(labels, dataset_name, **args)
    elif task in ['canny', 'lineart', 'hed']:
        return edge_label_transform(labels, dataset_name, **args)
    else:
        raise NotImplementedError("Only support segmentation and edge detection for now.")


def group_random_crop(images, resolution):
    """
    Args:
        images (list of PIL Image or Tensor): List of images to be cropped.

    Returns:
        List of PIL Image or Tensor: List of cropped image.
    """

    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    for idx, image in enumerate(images):
        i, j, h, w = RandomCrop.get_params(image, output_size=resolution)
        images[idx] = F.crop(image, i, j, h, w)

    return images


norm_layer = nn.InstanceNorm2d
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class LineDrawingModel(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, n_residual_blocks=3, sigmoid=True):
        super(LineDrawingModel, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out



class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HEDdetector(nn.Module):
    def __init__(self, model_path = None):
        super().__init__()
        if model_path is not None:
            state_dict = torch.hub.load_state_dict_from_url(model_path, map_location=torch.device('cpu'))

            self.netNetwork = ControlNetHED_Apache2()
            self.netNetwork.load_state_dict(state_dict)
        else:
            self.netNetwork = ControlNetHED_Apache2()

    def __call__(self, input_image):
        H, W = input_image.shape[2], input_image.shape[3]

        edges = self.netNetwork((input_image * 255).clip(0, 255))
        edges = [torch.nn.functional.interpolate(edge, size=(H, W), mode='bilinear') for edge in edges]
        edges = torch.stack(edges, dim=1)
        edge = torch.sigmoid(torch.mean(edges, dim=1))
        
        edge = (edge * 255.0).clip(0, 255).to(torch.uint8)

        return edge / 255.0


class KeypointLoss(KeypointMSELoss):
    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:

        _mask = self._get_mask(target, target_weights, mask)
        if _mask is None:
            loss = torch.nn.functional.mse_loss(output, target, reduction='none')
        else:
            _loss = torch.nn.functional.mse_loss(output, target, reduction='none')
            loss = _loss * _mask

        loss = loss.mean(dim=(1,2,3))
        return loss * self.loss_weight

class DisplacementLoss(SoftWeightSmoothL1Loss):
    def forward(self, output, target, target_weight=None):

        if self.use_target_weight:
            assert target_weight is not None
            assert output.ndim >= target_weight.ndim

            for _ in range(output.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)

            loss = self.criterion(output, target) * target_weight

            if self.supervise_empty:
                loss = loss.mean(dim=(1,2,3))
            else:
                loss = loss.sum(dim=(1,2,3))
                target_weight = torch.threshold((target_weight>0).sum(dim=(1,2,3)), 1, 1)
                loss = loss / target_weight
                # num_elements = torch.nonzero(target_weight > 0).size()[0]
                # loss = loss.sum() / max(num_elements, 1.0)
        else:
            loss = self.criterion(output, target, reduction='none')
            loss = loss.mean(dim=(1,2,3))

        return loss * self.loss_weight