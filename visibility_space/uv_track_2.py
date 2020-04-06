import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotBL

"""
Projetcted baseline changing with time
"""

# Observation parameters
c = 3e8                                         # Speed of light
f = 1420e9                                      # Frequency
lam = c/f                                       # Wavelength
dec = (np.pi/180)*(-74-39./60-37.34/3600)       # Declination

time_steps = 600                                # Time steps
h = np.linspace(0,24,num=time_steps)*np.pi/12   # Hour angle window

# Generation of one baseline from 2 antennas ant1 and ant2
ant1 = np.array([25.095, -9.095, 0.045])
ant2 = np.array([90.284, 26.380, -0.226])
b_ENU = ant2-ant1
D = np.sqrt(np.sum(b_ENU**2))
L = (np.pi/180)*(-30-43./60-17.34/3600)

A = np.arctan2(b_ENU[0], b_ENU[1])
E = np.arcsin(b_ENU[2]/D)
print("Azimuth = " +str(A*180/np.pi))
print("Elevation = "+str(E*180/np.pi))


X = D*(np.cos(L)*np.sin(E) - np.sin(L)*np.cos(E)*np.cos(A))
Y = D*(np.cos(E)*np.sin(A))
Z = D*(np.sin(L)*np.sin(E) + np.cos(L)*np.cos(E)*np.cos(A))

u = lam**(-1)*(X*np.sin(h) + Y*np.cos(h))/1e3
v = lam**(-1)*(-X*np.sin(dec)*np.cos(h) + Y*np.sin(dec)*np.sin(h) + Z*np.cos(dec))/1e3
w = lam**(-1)*(X*np.cos(dec)*np.cos( h) - Y*np.cos(dec)*np.sin(h) + Z*np.sin(dec))/1e3

plotBL.UV(u,v,w)
plt.show()
