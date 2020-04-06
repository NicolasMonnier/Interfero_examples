import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotBL

## Generate the antennas position from the .txt file that contained physical position in ENU reference frame
antennaPosition = np.genfromtxt("configs/vlaa.enu.txt") # We're using Configuration from VLA 

# Plot the distributiton of the antennas
mxabs = np.max(np.abs(antennaPosition[:]))*1.1
fig = plt.figure(figsize=(6,6))
plt.plot((antennaPosition[:,0]-np.mean(antennaPosition[:,0]))/1e3, (antennaPosition[:,1]-np.mean(antennaPosition[:,1]))/1e3, 'o')
plt.axes().set_aspect('equal')
plt.xlim(-mxabs/1e3, mxabs/1e3)
plt.ylim(-mxabs/1e3, mxabs/1e3)
plt.xlabel("E (km)")
plt.ylabel("N (km)")
plt.title("antenna position")

## Observation parameters
c = 3e8             # Speed of light
f = 1420e6          # frequency
lam = c/f           # Wavelength

time_step = 1200    # Time steps
h = np.linspace(-6,6, num=time_step)*(np.pi/12)
# Declination convert in radian
L = np.radians(34.0790) # latitude of the VLA
dec = np.radians(34.)


# Interactive bar - change the number of Ntimes --> Time integration du to Earth's rotation
from ipywidgets import *
from IPython.display import display

def interplot(key, Ntimes):
    print("Ntimes = "+str(Ntimes))
    plotBL.plotuv(antennaPosition, L, dec, h, Ntimes, lam)

interplot("", 300)
plt.show()

