import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotBL

"""
Plot the uv-coverage of an EW-baseline whose filed ceenter
is at 2 different declination
"""

## Observation parameters
H = np.linspace(-6,6,600)*(np.pi/12)        # Hour angle in radian
d = 100 # We assume that the we already didived by the wavelength

delta = 60*(np.pi/180) # Declination in degrees
u_60 = d*np.cos(H)
v_60 = d*np.sin(delta)*np.sin(H)

## Simulating the sky 
RA_sources = np.array([5+30./60, 5+32./60+0.4/3600, 5+36./60+12.8/3600, 5+40./60+45.5/3600])
DEC_sources = np.array([60, 60+17./60+57./3600, 61+12./60+6.9/3600, 61+56./60+34./3600])
Flux_sources_label = np.array(["", "1 Jy", "0.5 Jy", "0.2 Jy"])
Flux_sources = np.array([1,0.5,0.2])
step_size = 200 

# Convertion of (alpha, delta) to (l,m)
RA_rad = np.array(RA_sources)*(np.pi/12)
DEC_rad = np.array(DEC_sources)*(np.pi/180)
RA_delta_rad = RA_rad - RA_rad[0]

l = np.cos(DEC_rad)*np.sin(RA_delta_rad)
m = np.sin(DEC_rad)*np.cos(DEC_rad[0]) - np.cos(DEC_rad)*np.sin(DEC_rad[0])*np.cos(RA_delta_rad)
print("l = ", l*(180/np.pi))
print("m = ", m*(180/np.pi))

point_sources = np.zeros((3,3))
point_sources[:,0] = Flux_sources
point_sources[:,1] = l[1:]
point_sources[:,2] = m[1:]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel("$l$ en degree")
plt.ylabel("$m$ en degree")
plt.plot(l[0], m[0], "bx")
plt.plot(l[1:]*(180/np.pi), m[1:]*(180/np.pi), "ro")

counter = 1
for xy in zip(l[1:]*(180/np.pi) + 0.25, m[1:]*(180/np.pi) + 0.25):
    ax.annotate(Flux_sources_label[counter], xy=xy, textcoords='offset points', horizontalalignment='right', verticalalignment='bottom')
    counter = counter +1
plt.grid()
#plt.show()

## Simulating an observation
# Creation of the uv-plane
u = np.linspace(-1*(np.amax(np.abs(u_60))) -10, np.amax(np.abs(u_60)) + 10, num=step_size, endpoint=True)
v = np.linspace(-1*(np.amax(np.abs(v_60))) -10, np.amax(np.abs(v_60)) + 10, num=step_size, endpoint=True)
uu, vv = np.meshgrid(u, v)
zz = np.zeros(uu.shape).astype(complex)

# We create the dimension of our visibility plane
s = point_sources.shape
for counter in range(1, s[0] +1):
    A_i = point_sources[counter - 1, 0]
    l_i = point_sources[counter - 1, 1]
    m_i = point_sources[counter - 1, 2]
    zz += A_i*np.exp(-2j*np.pi*(uu*l_i + vv*m_i))
zz = zz[:, ::-1]

# Let's compute the total visibilities for our simulated sky
u_track = u_60
v_track = v_60
z = np.zeros(u_track.shape).astype(complex)

s = point_sources.shape
for counter in range(1, s[0] +1):
    A_i = point_sources[counter-1, 0]
    l_i = point_sources[counter-1, 1]
    m_i = point_sources[counter-1, 2]
    z += A_i*np.exp(-2j*np.pi*(u_track*l_i + v_track*m_i))

# Sample our visibility plane on the uv-track derived in the 1st section
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(zz.real, extent=[-1*(np.amax(np.abs(u_60)))-10, np.amax(np.abs(u_60))+10, -1*(np.amax(np.abs(v_60)))-10, np.amax(np.abs(v_60))+10])
plt.plot(u_60, v_60, 'k')
plt.xlim([-1*(np.amax(np.abs(u_60)))-10, np.amax(np.abs(u_60))+10])
plt.ylim([-1*(np.amax(np.abs(v_60)))-10, np.amax(np.abs(v_60))+10])
plt.xlabel('u')
plt.ylabel('v')
plt.title("Real part of visibilities")

plt.subplot(122)
plt.imshow(zz.imag, extent=[-1*(np.amax(np.abs(u_60)))-10, np.amax(np.abs(u_60))+10, -1*(np.amax(np.abs(v_60)))-10, np.amax(np.abs(v_60))+10])
plt.plot(u_60, v_60, 'k')
plt.xlim([-1*(np.amax(np.abs(u_60)))-10, np.amax(np.abs(u_60))+10])
plt.ylim([-1*(np.amax(np.abs(v_60)))-10, np.amax(np.abs(v_60))+10])
plt.xlabel('u')
plt.ylabel('v')
plt.title("Imaginary part of visibilities")

# Now plot the visibilities as a function of time slot (ie -  V(u(ts), v(ts))
# The result is the real and imaginary part of the black curve plot previously, plotted as a function of time
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(z.real)
plt.xlabel('time-slot')
plt.ylabel('Jy')
plt.title('Real : sampled visibilities')

plt.subplot(122)
plt.plot(z.imag)
plt.xlabel('time-slot')
plt.ylabel('Jy')
plt.title('Imaginary : sampled visibilities')


# Doing the same for amplitude and phase
# For the visibility function :
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(np.abs(zz), extent=[-1*(np.amax(np.abs(u_60)))-10, np.amax(np.abs(u_60))+10, -1*(np.amax(np.abs(v_60)))-10, np.amax(np.abs(v_60))+10])
plt.plot(u_60, v_60, 'k')
plt.xlim([-1*(np.amax(np.abs(u_60)))-10, np.amax(np.abs(u_60))+10])
plt.ylim([-1*(np.amax(np.abs(v_60)))-10, np.amax(np.abs(v_60))+10])
plt.xlabel('u')
plt.ylabel('v')
plt.title("Amplitude of the visibilities")
plt.subplot(122)
plt.imshow(np.angle(zz), extent=[-1*(np.amax(np.abs(u_60)))-10, np.amax(np.abs(u_60))+10, -1*(np.amax(np.abs(v_60)))-10, np.amax(np.abs(v_60))+10])
plt.plot(u_60, v_60, 'k')
plt.xlim([-1*(np.amax(np.abs(u_60)))-10, np.amax(np.abs(u_60))+10])
plt.ylim([-1*(np.amax(np.abs(v_60)))-10, np.amax(np.abs(v_60))+10])
plt.xlabel('u')
plt.ylabel('v')
plt.title("Phase of the visibilities")

# for the visibility function sampled by the black curve , plotted as function of time
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(abs(z))
plt.xlabel('time-slot')
plt.ylabel('Jy')
plt.title('Abs : sampled visibilities')

plt.subplot(122)
plt.plot(np.angle(z))
plt.xlabel('time-slot')
plt.ylabel('Jy')
plt.title('Phase : sampled visibilities')



plt.show()





