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
    
    upright_output = np.load(depth_dir + str(i) + '_corrected_output.npy') 
    rot_output = np.load(depth_dir + str(i) + '_rot_output.npy')
    input_output = np.load(depth_dir + str(i) + '_input_output.npy')
    
    gt = cv2.imread(rgb_dir + data_info['input'][i].replace('color','depth') + '.exr', cv2.IMREAD_ANYDEPTH) 
    gt[gt > 40] = 0    

   
    plt.subplot(1,3,1)
    plt.axis('off')
    plt.imshow(rgb)
    plt.subplot(1,3,2)
    plt.axis('off')
    plt.imshow(gt)
    plt.subplot(1,3,3)
    plt.axis('off')
    plt.imshow(input_output)
    plt.savefig('rectified_vs_rotated_depth_1.png', bbox_inches='tight')
    plt.subplot(1,3,1)
    plt.axis('off')
    plt.imshow(rgb_random_rot)
    plt.subplot(1,3,2)
    plt.axis('off')
    plt.imshow(synthesizeRotation(gt, random_r.as_matrix()))
    plt.subplot(1,3,3)
    plt.axis('off')
    plt.imshow(rot_output)
    plt.savefig('rectified_vs_rotated_depth_2.png', bbox_inches='tight')

if __name__ == '__main__':
    create_data(int(sys.argv[1]))
