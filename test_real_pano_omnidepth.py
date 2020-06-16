import torch
import torch.nn as nn
import util

from omnidepth_trainer import OmniDepthTrainer
from network import *
from dataset import *
from util import mkdirs, set_caffe_param_mult
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from pdb import set_trace as pause

import os.path as osp
import timeit 

from fibonacci_sph import fibonacci_sphere, cart2sph


# --------------
# PARAMETERS
# --------------
num_points = 100
num_tests = 10000
seed = 42
network_type = 'RectNet' # 'RectNet' or 'UResNet'
experiment_name = 'omnidepth'
input_dir = '/home/paulo/datasets/3d60' # Dataset location
val_file_list = './splits/my_test.txt' # List of evaluation files
checkpoint_dir = osp.join('experiments', experiment_name)
checkpoint_path = 'rectnet.pth'
# checkpoint_path = osp.join(checkpoint_dir, 'checkpoint_latest.pth')
num_workers = 4
validation_sample_freq = -1
device_ids = [0]
#input_path = '/home/paulo/datasets/lab/lab_original/SAM_100_0130.jpg'
#input_path = '/home/paulo/datasets/3d60/Matteport3D/92_fa5f164b48f043c6b2b0bb9e8631a4821_color_0_Left_Down_0.0.png'

# -------------------------------------------------------
# Fill in the rest
env = experiment_name
device = torch.device('cuda', device_ids[0])

# UResNet
if network_type == 'UResNet':
	model = UResNet()
	alpha_list = [0.445, 0.275, 0.13]
	beta_list = [0.15, 0., 0.]
# RectNet
elif network_type == 'RectNet':
	model = RectNet()
	alpha_list = [0.535, 0.272]
	beta_list = [0.134, 0.068,]
else:
	assert True, 'Unsupported network type'

# Make the checkpoint directory
mkdirs(checkpoint_dir)


# -------------------------------------------------------
# Set up the training routine
network = nn.DataParallel(
	model.float(),
	device_ids=device_ids).to(device)


val_dataloader = torch.utils.data.DataLoader(
        dataset=OmniDepthDataset(
                root_path=input_dir,
                path_to_img_list=val_file_list),
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False)

trainer = OmniDepthTrainer(
        experiment_name,
        network,
        None,
        val_dataloader,
        None,
        None,
        checkpoint_dir,
        device,
        validation_sample_freq=validation_sample_freq)


points = fibonacci_sphere(num_points)

trainer.evaluate_pano(checkpoint_path)
