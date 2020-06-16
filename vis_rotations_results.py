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
    plt.imshow(pano)
    plt.show()
    panos = np.zeros((samples,)+pano.shape)
    r = R.from_rotvec([0, 0, 0])
    points = fibonacci_sphere(samples)
    for i in range(len(points)):
        az, el, _ = cart2sph(points[i][0], points[i][1], points[i][2])
        panos[i] = synthesizeRotation(pano, R.from_rotvec([el, 0, az]).as_matrix())
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    for point in points:
        ax.scatter(point[0], point[1], point[2], c='r', marker='o')
    plt.show()
    
 
    results = pd.read_csv('rotations_results.txt')
    error = np.zeros(100)
    frequency = np.zeros(100)
    for index, row in results.iterrows():    
        error[int(row['point'])] += row['abs_rel_error']
        frequency[int(row['point'])] += 1
    error = error / frequency

    fig = plt.figure(figsize=(20,10))
    for idx in range(np.shape(panos)[0]):
        if idx % 4 == 0:
            plt.subplot(samples/20, 5, (idx+4)/4)
            plt.axis('off')
            plt.imshow(panos[idx])
            plt.title('{0:.4f}'.format(error[idx]))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    fig.savefig('rotations_results.png')

