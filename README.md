# [NeurIPS 2025]: CPO: Condition Preference Optimization for Controllable Image Generation
<div align="center">

  [Zonglin Lyu](https://zonglinl.github.io/), [Ming Li](https://liming-ai.github.io/), [Xinxin Liu](https://openreview.net/profile?id=~Xinxin_Liu4), [Chen Chen](https://www.crcv.ucf.edu/chenchen/)

  [![Website shields.io](https://img.shields.io/website?url=http%3A//poco.is.tue.mpg.de)](https://zonglinl.github.io/CPO_page/) [![YouTube Badge](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://youtu.be/r6jX43e1EGI)  [![arXiv](https://img.shields.io/badge/arXiv-2507.04984-00ff00.svg)](TODO)
  
</div>


<p align="center">
<img src="images/Teaser.png" width=95%>
<p>

## Overview
We introduce Condition Preference Optimization to directly contrasting input conditions instead of raw images for controllable image generation task.

<p align="center">
<img src="images/overview.png" width=95%>
<p>

## Quantitative Results
Our method achieves state-of-the-art performance in controllability among all recent SOTAs and does not introduce noise to image quality during training. 
<p align="center">
<img src="images/Controllability.png" width=95%>
<p>

<p align="center">
<img src="images/FID.png" width=95%>
<p>


We employ the DINO-v2 adapter proposed in ControlAR and reveals that it is beneficial to ControlNet-based methods in both controllability and image quality.
<p align="center">
<img src="images/DINO-Controllability.png" width=95%>
<p>

<p align="center">
<img src="images/DINO-FID.png" width=95%>
<p>


## Qualitative Results
Our method achieves the best qualitative visualization.
<p align="center">
<img src="images/Qual1.png" width=95%>
<p>

For more visualizations, please refer to our <a href="https://zonglinl.github.io/CPO_page/">project page</a>.


## Environment setup

```bash
conda create -n reward python=3.11 -y
conda activate reward
```

```bash
bash ./prepare_env.sh
```


## Usage

### Training

#### Original ControlNet


```bash
python download_controlnetpp.py ## download ControlNet++ 
sh train/train_pose.sh ## finetune original ControlNet only for pose.
sh train/reward_[task_name]_CPO.sh ## train CPO

```


#### ControlNet with DINO adapter

```bash
sh train/reward_[task_name].sh ## use train/reward_DINO.py for dino adapter finetuning in the script.
# If you want to add ControlNet++ training, you can switch to train/reward_control.py and resume pretrained ckpt.
# reward_control.py is currently default to ControlNet-DINO. If you want to train original ControlNet++ please import ControlNetModel from diffusers


sh train/reward_[task_name]_CPO_DINO.sh ## train CPO for ControlNet-DINO

```


### Evaluation

```bash
eval/eval_{task}.sh # You need to switch eval.py to eval_dino.py when you want to evaluate the ControlNet-Dino.
eval/eval_{task}_flux_cpo.sh # We currently only support Lineart FLUX-ControlNet. 
```

## Model Checkpoints


|Model| Lineart | Canny | Hed | Depth | Seg (ADE20K) | Seg (CoCo) | Pose |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| ControlNet |  [link](https://huggingface.co/ucfzl/ControlNet_Lineart_CPO) | [link](https://huggingface.co/ucfzl/ControlNet_Canny_CPO) | [link](https://huggingface.co/ucfzl/ControlNet_Hed_CPO) |  [link](https://huggingface.co/ucfzl/ControlNet_Depth_CPO) | [link](https://huggingface.co/ucfzl/ControlNet_segmentation_ADE20K_CPO) |[link](https://huggingface.co/ucfzl/ControlNet_segmentation_COCOStuff_CPO) |[link](https://huggingface.co/ucfzl/ControlNet_Pose_CPO) |
| ControlNet-DINO |  [link](https://huggingface.co/ucfzl/ControlNet_DINO_Lineart_CPO) | [link](https://huggingface.co/ucfzl/ControlNet_DINO_Canny_CPO) | [link](https://huggingface.co/ucfzl/ControlNet_DINO_Hed_CPO) |  [link](https://huggingface.co/ucfzl/ControlNet_DINO_Depth_CPO) | [link](https://huggingface.co/ucfzl/ControlNet_DINO_segmentation_ADE20K_CPO) |[link](https://huggingface.co/ucfzl/ControlNet_DINO_segmentation_COCOStuff_CPO) |[link](https://huggingface.co/ucfzl/ControlNet_DINO_Pose_CPO) |
| ControlNet-FLUX |  [link](https://huggingface.co/ucfzl/FLUX_Lineart_CPO) | - | - |  - | - |- |- |

## Citation

If our work is helpful, please cite
```
@inproceedings{
lyu2025cpo,
title={{CPO}: Condition Preference Optimization for Controllable Image Generation},
author={Zonglin Lyu and Ming Li and Xinxin Liu and Chen Chen},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025}
}
```



