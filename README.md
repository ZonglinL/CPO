# [NeurIPS 2025]: CPO: Condition Preference Optimization for Controllable Image Generation

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
