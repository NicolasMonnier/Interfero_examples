import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# Read images
duck = mpimg.imread('figures/Anas_platyrhynchos_male_female_quadrat_512.png')

def rgb2gray(rgb):
    """
    Convert rgb image to gray image.
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b #standard grayscale conversion
    return gray

gDuck = rgb2gray(duck)


def circularSamplingMap(imgSize, outer, inner=0):
    """Return a circular sampling map of the size [imgSize, imgSize]
    imgSize : Size of the image in pixel
    outer   : outer radius (in pixel) to exclude sampling above
    inner   : inner radius (in pixel) to exclude sampling bellow
    """
    zero = np.zeros((imgSize, imgSize), dtype=float)
    one = np.ones((imgSize, imgSize), dtype=float)
    xpos, ypos = np.mgrid[0:imgSize, 0:imgSize] # Generate 2 grids of size imgSize*imgSize. The first grid contains the x pos and the 2nd the y pos
    radius = np.sqrt((xpos-imgSize/2)**2 + (ypos-imgSize/2)**2) # Add the 2 grids to get a single one which contain the distance from the central pixel 
    sampling = np.where((outer >= radius) & (radius >=inner), 1., 0.) # 
    return sampling


imgSize = 512

# Plot sammpling function and it's fourier transform (PSF)
sampling = circularSamplingMap(imgSize, 100, 30) # Get sampling function
fftSampling = (np.fft.fftshift(np.fft.ifft2(sampling))) # fft of the sampling function

fig, ax = plt.subplots(figsize=(16,8))
plt.subplot(1,2,1)
plt.title('Sampling function')
plt.imshow(sampling).set_cmap('gray')
plt.subplot(1,2,2)
plt.title('PSF')
plt.imshow(np.abs(fftSampling))
plt.colorbar(shrink=0.5)


# Observed image of the Duck
fftDuck = np.fft.fftshift(np.fft.fft2(gDuck))
observedDuck = np.fft.ifft2(np.fft.fftshift(sampling * fftDuck)) # Multiply sampling function and Fourier Transform of DUck then back to spatial domain
plt.figure(figsize=(8,8))
plt.title('Observed Duck')
plt.imshow(np.abs(observedDuck)).set_cmap('gray')

plt.show()
