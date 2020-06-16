import numpy as np
import cv2

from pdb import set_trace as pause

def synthesizeRotation(image, R):
    spi = SphericalImage(image)
    spi.rotate(R)
    return spi.getEquirectangular()

def getMapping(image, R):
    spi = SphericalImage(image)
    spi.rotate(R)
    return spi.getMap()

class SphericalImage(object):

    def __init__(self, equImage):
        self.__colors = equImage
        self.__dim = equImage.shape # height and width

        phi, theta = np.meshgrid(np.linspace(0, np.pi, num = self.__dim[0], endpoint=False), np.linspace(0, 2 * np.pi, num = self.__dim[1], endpoint=False))
        self.__coordSph = np.stack([(np.sin(phi) * np.cos(theta)).T,(np.sin(phi) * np.sin(theta)).T, np.cos(phi).T], axis=2)

    def rotate(self, R):
        data = np.array(np.dot(self.__coordSph.reshape((self.__dim[0]*self.__dim[1], 3)),R))
        self.__coordSph = data.reshape((self.__dim[0], self.__dim[1], 3))
        
        x, y, z = data[:,].T

        phi = np.arccos(z)
        theta = np.arctan2(y,x)
        theta[theta < 0] += 2*np.pi        
        theta = self.__dim[1]/(2*np.pi) * theta
        phi = self.__dim[0]/np.pi * phi

        self.mapped = np.stack([theta.reshape(256, 512), phi.reshape(256, 512)], axis=2).astype(np.float32)

        self.__colors = cv2.remap(self.__colors.cpu().data.numpy(), self.mapped, None, cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT)
        
    def getEquirectangular(self): return self.__colors
    def getSphericalCoords(self): return self.__coordSph
    def getMap(self): return self.mapped
