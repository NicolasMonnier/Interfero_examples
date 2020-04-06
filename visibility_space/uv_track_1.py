import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotBL

"""
Baseline projection as seen by the source
"""
# Generation of one baseline from 2 antennas ant1 and ant2
ant1 = np.array([-500e3, 500e3, 0]) # in m
ant2 = np.array([500e3, -500e3, +10])# in m

# Corresponding physical baseline in the ENU coordinates
b_ENU = ant2 - ant1             # baseline
D = np.sqrt(np.sum(b_ENU**2))   # |b|
print("|b| = " +str(D/1000)+" km")

# Lattitude of the interferometer is L = 45°00'00'' => Correspond au niveau de la France
L = (np.pi/180)*(45+0./60+0./3600)  # Lattitude in radian 

A = np.arctan2(b_ENU[0], b_ENU[1])
print("Baseline Azimuth  = "+str(np.degrees(A))+"°")
E = np.arcsin(b_ENU[2]/D)
print("Baseline Elevation = "+str(np.degrees(E))+"°")

plotBL.sphere(ant1, ant2, A, E, D, L)
plt.show()
