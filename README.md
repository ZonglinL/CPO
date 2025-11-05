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

First, download ControlNet++ ```download_controlnetpp.py```

Second, finetune original ControlNet with```train/reward_control_original.py``` for Pose only (```train/train_pose.sh``` for script)

Finally, train CPO with ```train/reward_CPO.py``` (```train/reward_[task_name]_CPO.sh``` for training script)

#### ControlNet with DINO adapter

First, finetune ControlNet with DINO adapter with ```train/reward_DINO.py``` (```train/reward_[task_name].sh``` for training script)

Second (optional), train ContolNet++ with DINO adapter with ```train/reward_control.py```. You will need to import ```ControlNetModel``` from ```controlnets/``` istead of ```diffusers```. This can be done in ```train/reward_[task_name].sh```.

Finally, train CPO after your pretrained models with  ```train/reward_CPO_DINO.py``` (```train/reward_[task_name]_CPO_DINO.sh``` for training script)
