pip3 install stable_baselines3[extra]
pip3 install torchvision
pip3 install opencv-python
pip3 install git+https://github.com/facebookresearch/segment-anything.git
bash dl_model.sh
wget -P models https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
