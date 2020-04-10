import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read images
cyclist = mpimg.imread('figures/Umberto_Boccioni_Dynamism_of_a_Cyclist_512.png')
duck = mpimg.imread('figures/Anas_platyrhynchos_male_female_quadrat_512.png')


def rgb2gray(rgb):
    """
    Convert rgb image to gray image.
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b #standard grayscale conversion
    return gray

gCyclist = rgb2gray(cyclist)
gDuck = rgb2gray(duck)

# FFT of both images
fftCyclist = np.fft.fftshift(np.fft.fft2(gCyclist))
fig, axes = plt.subplots(figsize=(16,8))
plt.suptitle('Fourier Transform of \'Dynamisme of a cyclist\' ')
plt.subplot(1,2,1)
plt.imshow(10.*np.log10(np.abs(fftCyclist))) # Amplitude (dB)
plt.subplot(1,2,2)
plt.imshow(np.angle(fftCyclist)) # Phase

fftDuck = np.fft.fftshift(np.fft.fft2(gDuck))
fig, axes = plt.subplots(figsize=(16,8))
plt.suptitle('Fourier Transform of Duck')
plt.subplot(1,2,1)
plt.imshow(10.*np.log10(np.abs(fftDuck)))
plt.subplot(1,2,2)
plt.imshow(np.angle(fftDuck))


# Plot Image with Cyclist's amplitude and Duck's phase
fig = plt.figure(figsize=(8,8))
plt.title('Hybrid Image of Cyclist (ampl.) and Duck (phase)')
phase = np.angle(fftDuck) # Phase of the Duck
ampl = np.abs(fftCyclist) # Amplitude of the Cyclist
fftHybrid = ampl*(np.cos(phase) + 1j*np.sin(phase))
Hybrid = np.abs(np.fft.ifft2(np.fft.fftshift(fftHybrid)))
img = plt.imshow(Hybrid)
img.set_cmap('gray')

fig = plt.figure(figsize=(8,8))
plt.title('Hybrid Image of Duck (ampl.) and Cyclist (phase)')
phase = np.angle(fftCyclist)
ampl = np.abs(fftDuck)
fftHybrid = ampl*(np.cos(phase) + 1j*np.sin(phase))
Hybrid = np.abs(np.fft.ifft2(np.fft.fftshift(fftHybrid)))
img = plt.imshow(Hybrid)
img.set_cmap('gray')


# Phase only
fig = plt.figure(figsize=(8,8))
plt.title('Duck (Phase only)')
phase = np.angle(fftDuck)
ampl = 1.*np.ones_like(fftDuck)
fftPhaseDuck = ampl*(np.cos(phase) + 1j*np.sin(phase))
phaseDuck = np.abs(np.fft.ifft2(np.fft.fftshift(fftPhaseDuck)))
img = plt.imshow(phaseDuck)
img.set_cmap('gray')


# Amplitude only 
fig, axes = plt.subplots(figsize=(16,8))
plt.title('Duck (Amplitude Only)')
phs = np.zeros_like(fftDuck)
ampl = np.abs(fftDuck)
fftAmpDuck = ampl

plt.subplot(1,2,1)
AmpDuck = np.abs(np.fft.fftshift(np.fft.ifft2(fftAmpDuck)))
img = plt.imshow(AmpDuck)
img.set_cmap('gray')

plt.subplot(1,2,2)
AmpDuck = 10.*np.log10(np.abs(np.fft.ifftshift(np.fft.ifft2(fftAmpDuck))))
img = plt.imshow(AmpDuck)
img.set_cmap('gray')


plt.show()

