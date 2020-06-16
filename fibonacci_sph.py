import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PanoProcessing.rotate_pano import synthesizeRotation
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
from pdb import set_trace as pause

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def fibonacci_sphere(samples=1):
    points = np.zeros((samples,3))
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = (i % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points[i] = [x,y,z]

    return points


if __name__=='__main__':
    samples = 100

    pano = mpimg.imread('/home/paulo/datasets/3d60/Matteport3D/92_fa5f164b48f043c6b2b0bb9e8631a4821_color_0_Left_Down_0.0.png')
     
    for i in range(10):
        for j in range(pano.shape[1]//2-5, pano.shape[1]//2+5):
            pano[i,j,0] = 255
            pano[i,j,1] = 0
            pano[i,j,2] = 0

    plt.imshow(pano)
    plt.show()
    panos = np.zeros((samples,)+pano.shape)
    r = R.from_rotvec([0, 0, 0])
    points = fibonacci_sphere(samples)
    sph_points = [cart2sph(p[0], p[1], p[2]) for p in points]

    for i in range(len(points)):
        az, el, _ = cart2sph(points[i][0], points[i][1], points[i][2])
        #panos[i] = synthesizeRotation(pano, R.from_euler('zyx',[az, 0, el]).as_matrix())
        panos[i] = synthesizeRotation(pano, R.from_rotvec(points[i]).as_matrix())
       
        
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    for i in range(samples):
        ax.scatter(points[i][0], points[i][1], points[i][2], c='r', marker='o')
        ax.text(points[i][0] + 0.01, points[i][1] + 0.01, points[i][2] + 0.01, "{0:.4f},{1:.4f}".format(np.rad2deg(sph_points[i][0]), np.rad2deg(sph_points[i][1])), fontsize=9)
        
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
    plt.show()
    
 

    fig = plt.figure(figsize=(20,10))
    for idx in range(np.shape(panos)[0]):

        az, el, _ = cart2sph(points[idx][0], points[idx][1], points[idx][2])
        
        plt.subplot(samples/10, 10, idx+1)
        plt.axis('off')
        plt.imshow(panos[idx])
            
        plt.title("{0:.4f},{1:.4f}".format(np.rad2deg(az),np.rad2deg(el)))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    fig.savefig('rotations.png')

