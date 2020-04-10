import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def pointSourceFFT(imgSize, ypos, xpos, amp=1):
    img = np.zeros((imgSize+1, imgSize+1)) # Odd size array so it can have a center
    img[ypos, xpos] = amp # Central pixel have an intensity of 1
    fftImg = np.fft.fft2(np.fft.fftshift(img)) 

    fig, axes = plt.subplots(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title('Image') 
    plt.imshow(img)
    plt.set_cmap('gray')
    plt.colorbar(shrink=0.5)

    plt.subplot(1,2,2)
    plt.title(' Fourier Transform - Phase')
    plt.imshow(np.angle(fftImg))
    plt.set_cmap('hsv')
    plt.colorbar(shrink=0.5)


    print("FFT max = ", np.max(np.abs(fftImg)), "FFT min = ", np.min(np.abs(fftImg)))



def pointSourceFFTCircle(imgSize, ypos, xpos, amp=1, rad=10):
    img = np.zeros((imgSize+1, imgSize+1))
    img[ypos, xpos] = amp
    fftImg = np.fft.fft2(np.fft.fftshift(img))

    fig, ax = plt.subplots(figsize=(16,8))
    ax = plt.subplot(1,2,1)
    plt.title('Image')
    c = plt.Circle(((imgSize/2)+1, (imgSize/2)+1), rad, color='b', linewidth=1, fill=False)
    ax.add_patch(c)
    plt.imshow(img, interpolation='nearest')
    plt.set_cmap('gray')
    plt.colorbar(shrink=0.5)

    plt.subplot(1,2,2)
    plt.title('Fourier Transform - Phase')
    plt.imshow(np.angle(fftImg))
    plt.set_cmap('hsv')
    plt.colorbar(shrink=0.5)


def multipleSourceFFT(imgSize, pos, amp):
    img = np.zeros((imgSize+1, imgSize+1))
    for p, a in zip(pos, amp):
        img[p[0], p[1]] = a
        print(1)
    fftImg = np.fft.fft2(np.fft.fftshift(img))

    fig, ax = plt.subplots(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title('Image')
    plt.imshow(img)
    plt.set_cmap('gray')
    plt.colorbar(shrink=0.5)

    plt.subplot(1,2,2)
    plt.title('Fourier Transform - Phase')
    plt.imshow(np.angle(fftImg))
    plt.set_cmap('hsv')
    plt.colorbar(shrink=0.5)


def reconstructImage(vis, nbsamples):
    """Randomly select a few number of samples from the spatial frequency domain (visibilities)
    and reconstruct the image with those samples. To do a full reconstruction of the image, the
    number of samples has to be higher than the total number of pixels in the image because the
    np.random.randint choose with replacement, so position will be double counted
    """
    subVis = np.zeros_like(vis)
    ypos = np.random.randint(0, vis.shape[0] -1, size=int(nbsamples)) 
    xpos = np.random.randint(0, vis.shape[1] -1, size=int(nbsamples))
    subVis[ypos, xpos] = vis[ypos, xpos] #  Insert the random visibilities to the subset

    newImg = np.abs(np.fft.ifft2(np.fft.fftshift(subVis)))

    fig, ax = plt.subplots(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title('Sampled visibilities')
    plt.imshow(np.abs(subVis).astype(bool))
    plt.set_cmap('gray')

    plt.subplot(1,2,2)
    plt.title('Reconstructed Image')
    plt.imshow(newImg)
    plt.set_cmap('gray')



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

fftDuck = np.fft.fftshift(np.fft.fft2(gDuck))
reconstructImage(fftDuck, 1)
reconstructImage(fftDuck, 1e2)
reconstructImage(fftDuck, 1e3)
reconstructImage(fftDuck, 1e4)
reconstructImage(fftDuck, 1e5)
reconstructImage(fftDuck, 1e6)


imgSize = 128
#pointSourceFFT(128, 65-40, 65)
#pointSourceFFTCircle(128, 65,55)
#multipleSourceFFT(128, [[64,65], [90,65]], [1.,1.])


plt.show()
