import torch
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__)
depth_anything_path = os.path.join(current_dir, '..', 'Depth-Anything-V2')
sys.path.append(depth_anything_path)

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

def load_depth_model(encoder='vits', ckpt_filename='depth_anything_v2_vits.pth'):
    
    ckpt_path = os.path.join(current_dir, ckpt_filename)

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return model.to(DEVICE).eval()

def infer_depth(model, image_bgr):
    return model.infer_image(image_bgr)

def save_depth_map(depth_map, save_path="output/depth_map.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    plt.imsave(save_path, depth_norm, cmap='inferno')
    print(f"Depth map saved: {save_path}")