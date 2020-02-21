import numpy as np
import matplotlib.pyplot as plt


# Function that solve normal equations
def solve_normal(b, A, W):
	ATW = np.dot(A.T.conj(), W)

	return np.linalg.inv(ATW.dot(A)).dot(ATW.dot(b))


# Simulate some data 
# f(t) = c + mt

m = -2.5 #Model slope
c = 10.0 #Model y-intercept
N = 15   #Number of data point
xi = 10*np.random.random(N) #Random points in the domain (0, 10)
x = np.linspace(0, 10, N)

y = c + m*x # The real model 
deltay = 5*np.abs(np.random.random(N)) #Uncertainties in the data

Sigma = np.diag(deltay**2) #Covariance Matrix
W = np.diag(1.00/deltay**2) # The weight matrix

epsilon = np.random.multivariate_normal(np.zeros(N), Sigma) # A realisation of the noise
yi = c + m*xi + epsilon # The data point

# Plot the model and the data
plt.figure('Linear Model', figsize=(15,7))
plt.plot(x, y, label=r'$true model$')
plt.errorbar(xi, yi, deltay, fmt='xr', label=r'$ Data With 1-Sigma Uncertainty$')
plt.legend()


# Construct the desgin matrix
A = np.ones([N,2])
A[:,1] = xi 

AM = np.ones([N,2])
AM[:,1] = x

# Solve normal equations without accounting for uncertainty
xbar_no_uncertainty = solve_normal(yi, A, np.eye(N))
print("Best fit parameters without  accounting for uncertainty c = %f, m = %f"%(xbar_no_uncertainty[0], xbar_no_uncertainty[1] ))

# Reconstruct the fonction corresponding to these parameter
y_no_uncertainty = np.dot(AM, xbar_no_uncertainty)

# Solve the normal equation while accounting uncertainty
xbar = solve_normal(yi, A, W)
print("Best fit parameters while accounting for uncertainty c = %f, m = %f"%(xbar[0], xbar[1]))

# Reconstruct the function corresponding to these parameters
y_with_uncertainty = np.dot(AM, xbar)

plt.plot(x, y_no_uncertainty, 'g', label='$ Best model without uncertainty$')
plt.plot(x, y_with_uncertainty, 'm', label = '$ Best model with uncertainty$')

plt.legend()
plt.savefig("Linear_Regression.png")
