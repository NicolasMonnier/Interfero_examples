import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from astropy.io import fits
#import aplpy

#Disable astropy/aplpy logging
import logging
logger0 = logging.getLogger('astropy')
logger0.setLevel(logging.CRITICAL)
logger1 = logging.getLogger('aplpy')
logger1.setLevel(logging.CRITICAL)


def subtractPSF(img, psf, l, m, flux, gain):

    """Subtract the PSF (attenuated by the gain and flux) centred at (l,m) from an image"""

    #get the half lengths of the PSF
    if (psf.shape[0] % 2 == 0): psfMidL = int(psf.shape[0]/2) #even
    else: psfMidL = int((psf.shape[0]+1)/2) #odd
    if (psf.shape[1] % 2 == 0): psfMidM = int(psf.shape[1]/2) #even
    else: psfMidM = int((psf.shape[1]+1)/2) #odd

    #determine limits of sub-images
    #starting m
    if m-psfMidM < 0:
        subM0 = 0
        subPSFM0 = psfMidM-m
    else:
        subM0 = m-psfMidM
        subPSFM0 = 0
    #starting l
    if l-psfMidL < 0:
        subL0 = 0
        subPSFL0 = psfMidL-l
    else:
        subL0 = l-psfMidL
        subPSFL0 = 0
    #ending m
    if img.shape[1] > m+psfMidM:
        subM1 = m+psfMidM
        subPSFM1 = psf.shape[1]
    else:
        subM1 = img.shape[1]
        subPSFM1 = psfMidM + (img.shape[1]-m)
    #ending l
    if img.shape[0] > l+psfMidL:
        subL1 = l+psfMidL
        subPSFL1 = psf.shape[0]
    else:
        subL1 = img.shape[0]
        subPSFL1 = psfMidL + (img.shape[0]-l)

    #select subset of image
    #subImg = img[subL0:subL1, subM0:subM1]
    #select subset of PSF
    subPSF = psf[subPSFL0:subPSFL1, subPSFM0:subPSFM1]

    #subtract PSF centred on (l,m) position
    img[subL0:subL1, subM0:subM1] -= flux * gain * psf[subPSFL0:subPSFL1, subPSFM0:subPSFM1]
    return img


def gauss2D(x, y, amp, meanx, meany, sigmax, sigmay):
    """2D Gaussian Function"""
    gx = -(x - meanx)**2/(2*sigmax**2)
    gy = -(y - meany)**2/(2*sigmay**2)

    return amp * np.exp( gx + gy)

def err(p, xx, yy, data):
    """Error function for least-squares fitting"""
    return gauss2D(xx.flatten(), yy.flatten(), *p) - data.flatten()

def idealPSF(psfImg):
    """Determine the ideal PSF size based on the observing PSF doing a simple 2D Gaussian least-squares fit"""
    xx, yy = np.meshgrid(np.arange(0, psfImg.shape[0]), np.arange(0, psfImg.shape[1]))
    # Initial estimate: PSF should be amplitude 1, and usually imaging over sample the PSF by 3-5 pixels
    params0 = 1., psfImg.shape[0]/2., psfImg.shape[1]/2., 3., 3.
    params, pcov, infoDict, errmsg, sucess = optimize.leastsq(err, params0, \
                            args=(xx.flatten(), yy.flatten(), psfImg.flatten()), full_output=1)
    #fwhm = [2.*np.sqrt(2.*np.log(2.)) * params[3], 2.*np.sqrt(2.*np.log(2.)) * params[4]]
    return params


def restoreImg(skyModel, residImg, params):
    """Generate a restored image from a deconvolved sky model, residual image, ideal PSF params"""
    mdlImg = np.zeros_like(residImg)
    for l,m, flux in skyModel:
        mdlImg[l,m] += flux
    
    #generate an ideal PSF image
    psfImg = np.zeros_like(residImg)
    xx, yy = np.meshgrid(np.arange(0, psfImg.shape[0]), np.arange(0, psfImg.shape[1]))
    psfImg = gauss2D(xx, yy, params[0], params[1], params[2], params[3], params[4])
    
    #convolve ideal PSF with model image
    sampFunc = np.fft.fft2(psfImg) #sampling function
    mdlVis = np.fft.fft2(mdlImg) #sky model visibilities
    sampMdlVis = sampFunc * mdlVis #sampled sky model visibilities
    convImg = np.abs(np.fft.fftshift(np.fft.ifft2(sampMdlVis))) + residImg #sky model convolved with PSF
    
    #return mdlImg + residImg
    return convImg



def hogbom(dirtyImg, psfImg, gain, niter, fthresh):
    """Implementation of Hogbom's CLEAN method
    inputs:
    dirtyImg: 2-D numpy array, dirty image
    psfImg: 2-D numpy array, PSF
    gain: float, gain factor, 0 < gain < 1
    niter: int, maximum number of iterations to halt deconvolution
    fthresh: flaot, maximum flux threshold to halt deconvolution
    
    outputs:
    residImg: 2-D numpy array, residual image
    skyModel: list of sky model components, each component is a list [l, m, flux]
    """
    #Initalization
    skyModel = [] #initialize empty model
    residImg = np.copy(dirtyImg) #copy the dirty image to the initial residual image
    i = 0 #number of iterations
    print('\tMinor Cycle: fthresh: %f'%fthresh)
    
    #CLEAN iterative loop
    while np.max(residImg) > fthresh and i < niter:
        print("Minor Cycle n°",i)
        lmax, mmax = np.unravel_index(residImg.argmax(), residImg.shape) #get pixel position of maximum value
        fmax = residImg[lmax, mmax] #flux value of maximum pixel
        #print 'iter %i, (l,m):(%i, %i), flux: %f'%(i, lmax, mmax, fmax)
        residImg = subtractPSF(residImg, psfImg, lmax, mmax, fmax, gain)
        skyModel.append([lmax, mmax, gain*fmax])
        i += 1
    
    return residImg, skyModel

def selectSubPSF(psfImg):
    """Select out a central region of the PSF for the Hogbom minor cycle,
    and compute the ratio between the main lobe and highest sidelobe

    There are a number of ways to implement this function. A general method
    would determine where the first sidelobe is, and select out a region of
    the PSF which includes those sidelobes. A simple method would be to hard-code
    the size of the sub-image. Marks will be given based on how general the
    function is.

    inputs:
    psfImg: 2-D numpy array, PSF

    outputs:
    subPsfImg: 2-D numpy array, subset of the PSF image which contains the main lobe out to the
        largest sidelobe.
    peakRatio: float, peakRatio < 1, ratio of the largest sidelobe to the main lobe
    """

    # We first want to get the center of the PSF
    if (psfImg.shape[0] % 2 == 0): psfMidL = int(psfImg.shape[0]/2) #even
    else: psfMidL = int((psfImg.shape[0]+1)/2) #odd
    if (psfImg.shape[1] % 2 == 0): psfMidM = int(psfImg.shape[1]/2) #even
    else: psfMidM = int((psfImg.shape[1]+1)/2) #odd
 
    psf_max_peak   = psfImg[psfMidL, psfMidM]
    lobes_max_peak = 0
    lobes_max_pos  = 0
    
    end_lobe    = False
    start_lobe  = False
    curr_peak   = psfImg[psfMidL, psfMidM]
    prev_peak   = curr_peak

    # Crappy version, We look point after point, in one direction (considering it's symetric) the value of the pixel then stop when the 1st sidelobes is reached
    for i in range(psfMidL):
        curr_peak = psfImg[psfMidL + i, psfMidM]
        if curr_peak > prev_peak:
            if start_lobe is False:
                start_lobe = True
            if end_lobe is True:
                lobes_max_pos = i
                break 
            lobes_max_peak = curr_peak
        else:
            if start_lobe is True:
                end_lobe = True
        prev_peak = curr_peak

    peakRatio = lobes_max_peak/psf_max_peak
    subPsfImg = np.copy(psfImg[psfMidL-lobes_max_pos:psfMidL+lobes_max_pos, psfMidM-lobes_max_pos:psfMidM+lobes_max_pos])

    #return sub-PSF and ratio
    return subPsfImg, peakRatio


def clark(dirtyImg, psfImg, gain, niter, fthresh):
    """Implementation of Clark's CLEAN method
    inputs:
    dirtyImg: 2-D numpy array, dirty image
    psfImg: 2-D numpy array, PSF
    gain: float, gain factor, 0 < gain < 1
    niter: int, maximum number of iterations to halt deconvolution
    fthresh: flaot, maximum flux threshold to halt deconvolution

    outputs:
    residImg: 2-D numpy array, residual image
    skyModel: list of sky model components, each component is a list [l, m, flux]
    """

    #Initalization
    skyModel = [] #initialize empty model
    residImg = np.copy(dirtyImg) #copy the dirty image to the initial residual image
    i = 0 #number of iterations

    #select subset of PSF
    subPsfImg, peakRatio = selectSubPSF(psfImg) #ADD CODE: this function needs to be written

    #CLEAN iterative Major cycle
    while np.max(residImg) > fthresh and i < niter:
        print("Major Cycle n°",i)

        #ADD CODE: get flux and pixel position of maximum value
        f_max = np.max(residImg)
        l_max, m_max = np.where(residImg == f_max)
        f_max = dirtyImg[l_max, m_max]

        #ADD CODE: minor cycle, run partial Hogbom, return partial sky model
        print(f_max*peakRatio)
        residImg, part_skyModel= hogbom(dirtyImg, subPsfImg, gain, 60, f_max*peakRatio)

        #ADD CODE: Fourier transform partial sky model to model visibilities
        sky = np.zeros_like(dirtyImg)
        for x, y, f in part_skyModel:
            sky[x, y] += f
        V_part_skyModel = np.fft.fft2(sky)

        #ADD CODE: compute sampling function from PSF
        V_sampling = np.fft.fft2(psfImg)

        #ADD CODE: subtract sampled model image from the residual image
        residImg = residImg - np.fft.ifft2(V_part_skyModel*V_sampling)

        #ADD CODE: update full sky model
        for x, y, f in part_skyModel:
            skyModel.append([x, y, f])


        print("Max resiImg = ",np.max(residImg))
        i += 1

    return residImg, skyModel



#input parameters
gain = 0.1 #loop gain, range: 0 < gain < 1
niter = 30 #number of loop iterations
fthresh = 2.5 #minimum flux threshold to deconvolve


#input images: dirty, PSF
#assuming unpolarized, single frequency image
fh = fits.open('data/KAT-7_6h60s_dec-30_10MHz_10chans_uniform_n100-dirty.fits')
dirtyImg = fh[0].data[0,0]
fh = fits.open('data/KAT-7_6h60s_dec-30_10MHz_10chans_uniform_n100-psf.fits')
psfImg = fh[0].data[0,0]
idealPSFparams = idealPSF(psfImg) #compute ideal PSF parameters

#plot the dirty image
fig = plt.figure(figsize=(8,8))
plt.imshow(psfImg)
plt.title('PSF')
plt.colorbar()

#run deconvolution
residImg, skyModel = clark(dirtyImg, psfImg, gain, niter, fthresh)

#plot the dirty image
fig = plt.figure(figsize=(8,8))
plt.imshow(dirtyImg)
plt.title('Dirty Image')
plt.colorbar()

#plot the residual image
fig = plt.figure(figsize=(8,8))
plt.imshow(abs(residImg))
plt.title('Residual Image')
plt.colorbar()

#plot the restored image
restImg = restoreImg(skyModel, residImg, idealPSFparams)
fig = plt.figure(figsize=(8,8))
plt.imshow(abs(restImg))
plt.title('Restored Image')
plt.colorbar()


fh = fits.open('data/KAT-7_6h60s_dec-30_10MHz_10chans_true.fits')
trueImg = fh[0].data[0,0]
fig = plt.figure(figsize=(8,8))
plt.imshow(trueImg)
plt.title('True Sky')
plt.colorbar()


plt.show()
