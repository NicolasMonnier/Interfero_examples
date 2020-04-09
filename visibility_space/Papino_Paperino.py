import numpy as np
import matplotlib.pyplot as plt


# Simulation of Orion
RA_sources = np.array([-(44+44/60+6.686/3600), -(44+44/60+6.686/3600)]) # Right Assension (alpha) of Center, Beltegeuse, Rigel 
DEC_sources = np.array([-(74+39/60+37.481/3600), -(73+39/60+37.298/3600)])      # Declination (delta)
Flux_sources = np.array([1, 0.2])

# Convertion to Radian
RA_rad = np.array(RA_sources)*(np.pi/12)
DEC_rad = np.array(DEC_sources)*(np.pi/180)
RA_delta_rad = RA_rad - RA_rad[0]

l = np.cos(DEC_rad)*np.sin(RA_delta_rad)
m = np.sin(DEC_rad)*np.cos(DEC_rad[0]) - np.cos(DEC_rad)*np.sin(DEC_rad[0])*np.cos(RA_delta_rad)

print("l = ", l)
print("m = ", m)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
plt.xlim([-1e6,1e6])
plt.ylim([-1e6,1e6])
plt.xlabel("$l$ en degree")
plt.ylabel("$m$ en degree")
plt.plot(l[0], m[0], "bx")
plt.plot(l[1:]*(180/np.pi), m[1:]*(180/np.pi), "ro")
plt.show()
