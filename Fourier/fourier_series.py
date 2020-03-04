import numpy as np
import matplotlib.pyplot as plt

# Let's consider the arbitraty function f(x) = (x - 0.5)³ + 3

# Define the range of the x-axis
x = (np.arange(1200) - 600)/200

# Calculate y as a function of x
#y = (x - 0.5)**3 + 3
y =  np.sin(4*x)
nmax = 50

def FourierSeriesApprox(xvals, yvals, nmax):
    approx = np.zeros_like(yvals)
    T = (xvals[-1] - xvals[0])
    w = 2*np.pi/T
    dt = xvals[1] - xvals[0]
    approx = approx + 1/T*(np.sum(yvals*dt))
    for t in range(len(xvals)):
        for n in (np.arange(nmax)+1):
            an = 2/T*np.sum(np.cos(w*n*xvals)*yvals)*dt
            bn = 2/T*np.sum(np.sin(w*n*xvals)*yvals)*dt
            approx[t] = approx[t] + an*np.cos(w*n*xvals[t]) + bn*np.sin(w*n*xvals[t])
    return approx

yApprox = FourierSeriesApprox(x, y, nmax)

# Plot the function
plt.figure(figsize=(18,12))
plt.plot(x, y, label="Arbitrary function")
plt.plot(x, yApprox, label="Fourier Series Approx")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.legend()
plt.show()
