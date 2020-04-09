import numpy as np
import matplotlib.pyplot as plt
import plotBL


# Antenna's position
antennas = np.array([[25.095, -9.095, 0.045],
                    [90.284, 26.380, -0.026],
                    [3.985, 26.893, 0.00],
                    [-21.605, 25.494, 0.019],
                    [-38.271, -2.592, 0.394],
                    [-61.595, -79.699, 0.702],
                    [-87.988, 75.754, 0.138]])

plt.scatter(antennas[:,0], antennas[:,1])
plt.grid('on')
plt.title("KAT-7 array layout")
#plt.show()

b12_ENU = antennas[1] - antennas[0]         # ENU baseline b12
D12 = np.sqrt(np.sum(b12_ENU**2))           # Length D12 of b12
    
A12 = np.arctan2(b12_ENU[0], b12_ENU[1])    # Azimuth angle of b21 in rad
E12 = np.arcsin(b12_ENU[2]/D12)

# Observation parameters
L = (-30+43/60+17.34/3600)*(np.pi/180)      # Lattitude in rad
f = 1.4e9
c = 3e8
H0 = -4 # Starting Hour angle 
H1 = 3  # Stopping Hour angle
nb_step = 600
H = np.linspace(H0, H1, nb_step)            # Hour angle window
delta = (-74+38/60+37.481/3600)*(np.pi/180)  # Field Center Declination
#delta = 0
alpha = (4+44/60+6.686/3600)*(np.pi/12)     # Field Center Assension

# Calculate X, Y, Z parameters
X12 = D12*(np.cos(L)*np.sin(E12) - np.sin(L)*np.cos(E12)*np.cos(A12))
Y12 = D12*(np.cos(E12)*np.sin(A12))
Z12 = D12*(np.cos(L)*np.sin(E12) - np.cos(L)*np.cos(E12)*np.cos(A12))

lamda = c/f     # Wavelength

# Res to calculate the ellipse TODO
res1 = np.sqrt(X12**2 + Y12**2)*lamda**(-1)
res2 = np.abs(np.sin(delta))*res1
res3 = delta*Z12*lamda**(-1)


# Generate coordinate (uv) pair for baseline 12 at H = -4h
u12 = lamda**(-1)*(np.sin(H)*X12 + np.cos(H)*Y12)/1e3
v12 = lamda**(-1)*(-np.sin(delta)*np.cos(H)*X12 + np.sin(delta)*np.sin(H)*Y12 + np.cos(delta)*Z12)/1e3

plotBL.UUVV(u12,v12)
plt.show()
