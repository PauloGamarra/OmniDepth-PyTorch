from numpy import *
from scipy import ndimage
from pdb import set_trace as pause

def synthesizeRotation(image, R):
    spi = SphericalImage(image)
    spi.rotate(R)
    return spi.getEquirectangular()

class SphericalImage(object):

    def __init__(self, equImage):
        self._colors = equImage.copy()
        self._dim = equImage.shape # height and width
        
          
        phi, theta = meshgrid(linspace(0, pi, num = self._dim[0], endpoint=False), linspace(0, 2 * pi, num = self._dim[1], endpoint=False))
        self._coordSph = stack([(sin(phi) * cos(theta)).T,(sin(phi) * sin(theta)).T,cos(phi).T], axis=2)
        
    def rotate(self, R):
        data = array(dot(self._coordSph.reshape((self._dim[0]*self._dim[1], 3)),R))
        self._coordSph = data.reshape((self._dim[0], self._dim[1], 3))
        
        x, y, z = data[:,].T

        phi = arccos(z)
        theta = arctan2(y,x)
        theta[theta < 0] += 2*pi        
        theta = self._dim[1]/(2*pi) * theta
        phi = self._dim[0]/pi * phi
       
        
        if len(self._dim) == 3:
            for c in range(self._dim[2]): self._colors[...,c] = ndimage.map_coordinates(self._colors[:,:,c], [phi, theta], order=1, prefilter=False, mode='reflect').reshape(self._dim[0],self._dim[1])
        else:
            self._colors = ndimage.map_coordinates(self._colors, [phi, theta], order=1, prefilter=False, mode='reflect').reshape(self._dim[0],self._dim[1])
    def getEquirectangular(self): return self._colors
    def getSphericalCoords(self): return self._coordSph

