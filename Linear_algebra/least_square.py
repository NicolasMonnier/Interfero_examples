import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import leastsq



def sinusoid(x, t):
	"""
	Return a vector containing the values of a sinusoid  with parameters x evaluated at point t

	INPUTs:
	t 	Value of independent variable at the sampled point
	x		Vector of parameters

	"""
	
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]

	return x1*np.sin(2*pi*x2*t + x3)


def sinusoid_jacobian(x, t):
	""" 
	Reurn the jacobian corresponding to the function defined in sinusoid

	INPUTs!:
	t	Value of independant variables at sampled point
	x 	vector of parameters
	"""

	x1 = x[0]
	x2 = x[1]
	x3 = x[3]

	jacobian = np.empty([t.shape[0], x.shape[0]])

	jacobian[:,0] = np.sin(2*pi*x2*t + x3)
	jacobian[:,1] = 2*pi*t*x1*np.cos(2*pi*x2*t + x3)
	jacobian[:,2] = x1*np.cos(2*pi*x2*t + x3)

	return jacobian


def sinusoid_residual(x, t, d):
	"""
	Return a vector containing  the residual values.

	INPUTs:
	x	Vector of parameters
	t	Value of independant variable at sampled point
	d	Vector of measured values
	"""

	return d - sinusoid(x, t)



