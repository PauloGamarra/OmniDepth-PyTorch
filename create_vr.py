import pandas as pd
import numpy as np 
from scipy.spatial.transform import Rotation as R 
from PanoProcessing.rotate_pano import synthesizeRotation 
import matplotlib.pyplot as plt 
import cv2
import matplotlib.image as mpimg
import sys

def create_data(i): 
    data_info = pd.read_csv('./upright_examples.txt') 
    depth_dir = './examples/' 
    rgb_dir = '/home/paulo/datasets/my3d60/' 
    rgb = mpimg.imread(rgb_dir + data_info['input'][i] + '.png') 
    random_r = R.from_euler('zyx', [0, data_info['rand_y'][i], data_info['rand_x'][i]], degrees=True) 
    correction_r = R.from_euler('zyx', [0, data_info['pred_y'][i], data_info['pred_x'][i]], degrees=True) 
 
    rgb_random_rot = synthesizeRotation(rgb, random_r.as_matrix()) 
    rgb_upright = synthesizeRotation(rgb_random_rot, correction_r.inv().as_matrix()) 
    
    pred_depth = np.load(depth_dir + str(i) + '_corrected_output.npy') 
    pred_depth_rot = synthesizeRotation(pred_depth, correction_r.as_matrix()) 
    
    gt = cv2.imread(rgb_dir + data_info['input'][i].replace('color','depth') + '.exr', cv2.IMREAD_ANYDEPTH) 
    gt[gt > 40] = 0    

   
    mpimg.imsave('./upright_vr/' + str(i) + '_rgb_random_rot.png', rgb_random_rot)
    mpimg.imsave('./upright_vr/' + str(i) + '_rgb_upright.png', rgb_upright)
    mpimg.imsave('./upright_vr/' + str(i) + '_rgb_original.png', rgb)
    np.save('./upright_vr/' + str(i) + '_pred_depth_rot.npy', pred_depth_rot)
    np.save('./upright_vr/' + str(i) + '_pred_depth_upright', pred_depth)
    np.save('./upright_vr/' + str(i) + '_gt_depth.npy', gt)

if __name__ == '__main__':
    create_data(int(sys.argv[1]))
