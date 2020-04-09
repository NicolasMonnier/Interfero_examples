import numpy as np
import matplotlib.pyplot as plt


# Simulation of Orion
RA_sources = np.array([5+30/60, 5+55/60+10.3053/3600, 5+14/60+32.272/3600]) # Right Assension (alpha) of Center, Beltegeuse, Rigel 
DEC_sources = np.array([0, 7+24/60+25.42/3600, -(8+12/60+5.898/3600)])      # Declination (delta)

# Convertion to Radian
RA_rad = np.array(RA_sources)*(np.pi/12)
DEC_rad = np.array(DEC_sources)*(np.pi/180)
RA_delta_rad = RA_rad - RA_rad[0]

l = np.cos(DEC_rad)*np.sin(RA_delta_rad)
m = np.sin(DEC_rad)*np.cos(DEC_rad[0]) - np.cos(DEC_rad)*np.sin(DEC_rad[0])*np.cos(RA_delta_rad)

print("l = ", l*(180/np.pi))
print("m = ", m*(180/np.pi))

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
plt.xlim([-9,9])
plt.ylim([-9,9])
plt.xlabel("$l$ en degree")
plt.ylabel("$m$ en degree")
plt.plot(l[0], m[0], "bx")
plt.plot(l[1:]*(180/np.pi), m[1:]*(180/np.pi), "ro")
plt.show()
