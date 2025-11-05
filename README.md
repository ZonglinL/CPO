# [NeurIPS 2025]: CPO: Condition Preference Optimization for Controllable Image Generation
<div align="center">

  [Zonglin Lyu](https://zonglinl.github.io/), [Ming Li](https://liming-ai.github.io/), [Xinxin Liu](https://openreview.net/profile?id=~Xinxin_Liu4), [Chen Chen](https://www.crcv.ucf.edu/chenchen/)

  [![Website shields.io](https://img.shields.io/website?url=http%3A//poco.is.tue.mpg.de)](https://zonglinl.github.io/CPO_page/) [![YouTube Badge](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://youtu.be/r6jX43e1EGI)  [![arXiv](https://img.shields.io/badge/arXiv-2507.04984-00ff00.svg)](TODO)
  
</div>


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
# reward_control.py is currently default to ControlNet-DINO. If you want to train ControlNet++ please import ControlNetModel from diffusers


sh train/reward_[task_name]_CPO_DINO.sh ## train CPO for ControlNet-DINO

```


### Evaluation

```bash
eval/eval_{task}.sh # You need to switch eval.py to eval_dino.py when you want to evaluate the ControlNet-Dino.
eval/eval_{task}_flux_cpo.sh # We currently only support Lineart FLUX-ControlNet. 
```

## Model Checkpoints

