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

# --------------
# PARAMETERS
# --------------
network_type = 'RectNet' # 'RectNet' or 'UResNet'
experiment_name = 'omnidepth'
input_dir = '' # Dataset location
val_file_list = '' # List of evaluation files
checkpoint_dir = osp.join('experiments', experiment_name)
checkpoint_path = 'rectnet.pth'
# checkpoint_path = osp.join(checkpoint_dir, 'checkpoint_latest.pth')
num_workers = -1
validation_sample_freq = -1
device_ids = [0]
input_path = '/home/paulo/datasets/lab/lab_original/SAM_100_0130.jpg'
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


#loading the model

checkpoint = torch.load(checkpoint_path)
util.load_partial_model(network, checkpoint['state_dict'])

#runnig inference

with torch.no_grad():
    pano_img = io.imread(input_path).astype(np.float32) / 255.
    pano = cv.resize(pano_img, (512,256))   
    pano = np.expand_dims(pano, axis=0)
    pano = np.rollaxis(pano, 3, 1)



    rgb_input = torch.from_numpy(pano).float().to(device)      
    rgb_input = [rgb_input]


    output = network(rgb_input)    

    depth = output[0].cpu().squeeze()
    depth2 = output[1].cpu().squeeze()

    plt.subplot(2,2,1)
    plt.imshow(pano_img)
    plt.subplot(2,2,3)
    plt.imshow(depth)
    plt.subplot(2,2,4)
    plt.imshow(depth2)
    plt.show()

    np.save('./omni_depth_pred' + osp.basename(input_path) + '.npy', depth.cpu())
