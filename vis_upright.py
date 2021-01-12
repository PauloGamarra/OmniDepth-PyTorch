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
    random_r = R.from_euler('zxy', [0, data_info['rand_x'][i], data_info['rand_y'][i]], degrees=True) 
    correction_r = R.from_euler('zxy', [0, data_info['pred_x'][i], data_info['pred_y'][i]], degrees=True) 
 
    rgb_random_rot = synthesizeRotation(rgb, random_r.as_matrix()) 
    rgb_upright = synthesizeRotation(rgb_random_rot, correction_r.inv().as_matrix()) 
    
    upright_output = np.load(depth_dir + str(i) + '_corrected_output.npy') 
    rot_output = np.load(depth_dir + str(i) + '_rot_output.npy')
    
    
    gt = cv2.imread(rgb_dir + data_info['input'][i].replace('color','depth') + '.exr', cv2.IMREAD_ANYDEPTH) 
    gt[gt > 40] = 0    


    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.figure(figsize = (16,64))
    gs1 = gridspec.GridSpec(1, 4)
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

 
    ax1 = plt.subplot(gs1[0])
    plt.axis('off')
    ax1.imshow(rgb_random_rot)
    ax1 = plt.subplot(gs1[1])
    plt.axis('off')
    ax1.imshow(synthesizeRotation(gt, random_r.as_matrix()))
    ax1 = plt.subplot(gs1[2])
    plt.axis('off')
    ax1.imshow(rot_output)
    ax1 = plt.subplot(gs1[3])
    plt.axis('off')
    ax1.imshow(synthesizeRotation(upright_output, correction_r.as_matrix()))

    
    plt.savefig('upright_vs_omnidepth.png', bbox_inches='tight')

    """
    plt.subplot(1,4,1)
    plt.axis('off')
    plt.imshow(rgb_random_rot)
    plt.subplot(1,4,2)
    plt.axis('off')
    plt.imshow(rot_output)
    plt.subplot(1,4,3)
    plt.axis('off')
    plt.imshow(synthesizeRotation(gt, random_r.as_matrix()))
    plt.subplot(1,4,4)
    plt.axis('off')
    plt.imshow(synthesizeRotation(upright_output, correction_r.as_matrix()))
    plt.savefig('upright_vs_omnidepth.png', bbox_inches='tight')
    plt.show()
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.tight_layout(pad=0.01)
    """
if __name__ == '__main__':
    create_data(int(sys.argv[1]))
