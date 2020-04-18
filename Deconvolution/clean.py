import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

import time
from astropy.io import fits
import matplotlib.cm as cm
from copy import copy


#  Creation of our Sky
imsize = 50
# Background noise
noise_rms = 0.1
I = noise_rms*(np.random.random([imsize,imsize])-0.5)
#Add 3 point sources with different flux value
I[20,20] += 1
I[32,15] += 1.45
I[30,34] += 1.12

# Create fake psf
PSFsize = 13
PSF = np.zeros((PSFsize,PSFsize))
PSFmid = int((PSFsize-1)/2)
PSF[:,PSFmid] = 0.5
PSF[PSFmid,:] = 0.5
d1, d2, = np.diag_indices_from(PSF)
PSF[d1, d2] = 0.5
PSF[d1,d2[::-1]] = 0.5
PSF[PSFmid-2:PSFmid+3,PSFmid-2:PSFmid+3] = 0
PSF[PSFmid-1:PSFmid+2,PSFmid-1:PSFmid+2] = 0.75
PSF[PSFmid,PSFmid] = 1.0

# Creation of our Dirty map : I_dirty = I_true°PSF
I_dirty = convolve2d(I, PSF, mode='same')

# Plot the 3 figures
fig, axes = plt.subplots(figsize=(16,16))
plt.subplot(1,3,1)
plt.imshow(I, cmap=cm.jet, interpolation='nearest')
plt.title("$I_{True}(l,m)$")
plt.subplot(1,3,2)
plt.imshow(PSF, cmap=cm.jet, interpolation='nearest')
plt.title("$PSF(l,m)$")
plt.subplot(1,3,3)
plt.imshow(I_dirty, cmap=cm.jet, interpolation='nearest')




# Högbom's Algo (Image-domain CLEAN)
# 1 - Make copy of the dirty Image caleld the residual Image
# 2 - Find the maximum pixel value and its position if the residual image
# 3 - Substract the PSF multiplied bu the peak pixel value and gain factor from the residual image at the position peak
# 4 - Record the position and magnitudeof the point source substracted in a model
# 5 - Go back to step 2, till threshold or iter is not reach
# 6 - Convolve the accumulated point source sky model with restoring beam 
# 7 - Add the remainder of the residual image
#
# Input         : Dirty Image; PSF
# Parameters    : Gain; Iteration limit; Flux threshold
# Output        : Sky model; residual image; restored image


# -----------------------------------------------------------
# Step 1
# ----------------------------------------------------------
I_res = copy(I_dirty)
# Set parameters
gain = 0.2
niter = 30
threshold = 5.*noise_rms

plotmax = np.max(I)
plotmin = np.min(I)
model = []

for i in range(niter):
    print("Iteration n°",i) 
    # -----------------------------------------------------------
    # Step 2
    # ----------------------------------------------------------
    max_flux = np.max(I_res)
    max_pos  = np.where(I_res == max_flux)

    # -----------------------------------------------------------
    # Step 3
    # ----------------------------------------------------------
    px, py = max_pos
    print(px)
    I_res[px[0]-PSFmid:px[0]+PSFmid+1, py[0]-PSFmid:py[0]+PSFmid+1] -= max_flux*gain*PSF
    print("Peak :", max_flux, " at Position (", px[0],",",py[0],")")

    # -----------------------------------------------------------
    # Step 4
    # ----------------------------------------------------------
    model.append([px[0], py[0], gain*max_flux])


    # -----------------------------------------------------------
    # Step 5
    # ----------------------------------------------------------
    if max_flux < threshold:
        print("Threshold reached at iteration ",i)
        break


f, ax = plt.subplots(1,2,figsize=(16,16))
plt.subplot(1,2,1)
plt.imshow(I_dirty, cmap=cm.jet, interpolation='nearest')
plt.title('$I_{Dirty}(l,m)$')
plt.subplot(1,2,2)
plt.imshow(I_res, cmap=cm.jet, vmax=plotmax, vmin=plotmin, interpolation='nearest')
plt.title('$I_{Res}(l,m)$')


# Dirty image and residual image, scaled on the residual image
plotmax = np.max(I_res)
plotmin = np.min(I_res)

fig, axes = plt.subplots(figsize=(16,6))

plt.subplot(121)
plt.title('$I^D(l,m)$')
plt.imshow(I_dirty, cmap=cm.jet, vmax=plotmax, vmin=plotmin, interpolation='nearest')
plt.colorbar()
plt.subplot(122)
plt.title('$I^R(l,m)$')
plt.colorbar()
plt.imshow(I_res, cmap=cm.jet, vmax=plotmax, vmin=plotmin, interpolation='nearest')

# True sky and deconvolve sky
I_model = np.zeros((imsize, imsize))
for x, y, f in model:
    I_model[x,y] += f

plotmax = np.max(I)
plotmin = np.min(I)
fig, axes = plt.subplots(figsize=(16,6))

plt.subplot(121)
plt.title('True Sky')
plt.imshow(I, cmap=cm.jet, vmax=plotmax, vmin=plotmin, interpolation='nearest')
plt.colorbar()

plt.subplot(122)
plt.title('Deconvolved Sky')
plt.imshow(I_model, cmap=cm.jet, vmax=plotmax, vmin=plotmin, interpolation='nearest')
plt.colorbar();


# first get just the main lobe of the star shaped PSF
main_lobe = np.zeros([PSFsize,PSFsize])
main_lobe[PSFmid-1:PSFmid+2,PSFmid-1:PSFmid+2] = 0.75
main_lobe[PSFmid,PSFmid] = 1.0

fig, axes = plt.subplots(figsize=(16,6))
plt.subplot(121)
plt.imshow(PSF, cmap=cm.jet, interpolation='nearest')
plt.colorbar()
plt.title('PSF(l,m)');
plt.subplot(122)
plt.imshow(main_lobe, cmap=cm.jet, interpolation='nearest')
plt.colorbar()
plt.title('main lobe(l,m)');


# Now fit a symetric 2D gaussian to the main lobe
import scipy.optimize as opt

def gaussian2dsymmetric(x,y, A, x0, y0, sigma):
    gauss2d = A*np.exp(-((x-x0)**2.0+(y-y0)**2.0)/(2*sigma**2.0))
    return gauss2d.ravel()

x,y = np.meshgrid(range(PSFsize), range(PSFsize))
popt, pvoc = opt.curve_fit(gaussian2dsymmetric,x,y, main_lobe.ravel(), p0=[1.0, 6.5, 6.5, 2.])
print(popt)
A, x0, y0, sigma = popt

clean_beam = gaussian2dsummetric((x,y), A, x0, y0, sigma).reshape(PSFsize,PSFsize)/A


plt.show()
