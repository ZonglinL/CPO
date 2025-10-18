pip3 install -r requirements.txt --no-dependencies

pip3 install -U "openmim==0.3.9"
mim install "mmengine== 0.10.6"
mim install "mmcv==2.1.0"
mim install "mmsegmentation==1.2.1"
mim install "mmdet==3.2.0"
mim install "mmpose==1.3.2"

pip3 install "clean-fid== 0.1.35"
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip3 install "flash-attn==2.7.4.post1"

pip3 install "image-reward==1.5"
pip3 install "qwen_vl_utils==0.0.10"
pip3 install "ultralytics==8.3.102"

pip3 install "numpy==1.26.4" 
