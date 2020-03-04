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

	return x1*np.sin(2*np.pi*x2*t + x3)


def sinusoid_jacobian(x, t):
	""" 
	Reurn the jacobian corresponding to the function defined in sinusoid

	INPUTs!:
	t	Value of independant variables at sampled point
	x 	vector of parameters
	"""

	x1 = x[0]
	x2 = x[1]
	x3 = x[2]

	jacobian = np.empty([t.shape[0], x.shape[0]])

	jacobian[:,0] = np.sin(2*np.pi*x2*t + x3)
	jacobian[:,1] = 2*np.pi*t*x1*np.cos(2*np.pi*x2*t + x3)
	jacobian[:,2] = x1*np.cos(2*np.pi*x2*t + x3)

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



def levenberg_marquardt(d, t, x, r_func, j_func, maxit=100, lamda=1, K=10, eps1=1e-6, eps2=1e-6):
    """
    Reurn a vector containing the optimal parameter values found by the algorithm.

    INPUTs:
    d       Vector of measured values.
    t       Value of independant variable at sampled points.
    x       Vector of parameters.
    r_func  Function which generate the residual vector.
    j_func  Function which generate the jacobian.
    maxiter Maximum number of iteration.
    lambda  Initial value of the tuning parameter.
    K       Initial value of retuning parameter.
    eps1    First tolerence parameter - Trigger when residual is below this number.
    eps2    Second tolerence parameter - Trigger when relative changes to the parameter vector are  below to this number
    """

    # Initialises some important values and stored the original lambda value.
    r = r_func(x, t, d)
    old_chi = np.linalg.norm(r)
    olamda = lamda
    it = 0

    while True:
        # Computes the parameter update
        # This is just the implementation of the mathematical update rule.

        J = j_func(x, t)
        JT = J.T
        JTJ = JT.dot(J)
        JTJdiag = np.eye(JTJ.shape[0])*JTJ
        JTJinv = np.linalg.pinv(JTJ + lamda*JTJdiag)
        JTr = JT.dot(r)

        delta_x = JTJinv.dot(JTr)
        x += delta_x

        # Convergence tests. If a solution has been found, returns the result.
        # The Chi value is the norm of the residual and is used to determine
        # Wether the solution is improving. If the Chi value is sufficiently 
        # small, the function terminates. The second test checks to see weather
        # or not the solution is improving, and terminates it if isn't.

        r = r_func(x, t, d)
        new_chi = np.linalg.norm(r)

        if new_chi < eps1:
            return x
        elif np.linalg.norm(delta_x) < eps2*(np.linalg.norm(x) + eps2):
            return x

        # Tuning stage. If the parameter update was good, continue and restore lamda.
        # If the update was bad, scale lamda by K and revert last update.

        if new_chi > old_chi:
            x -= delta_x
            lamda = lamda*K
        else:
            old_chi = new_chi
            lamda = olamda

        # If the number of iterations grows too large, return the last value of x.
        it += 1

        if it > maxit:
            return x
        
        



t = np.arange(-0.06, 0.06, 0.06/300)  # The points at which we will be taking our "measurement"
noise = 2*np.random.normal(size=(t.shape[0])) # A noise vector which we will use to manufacture "real" measurements.
true_x = np.array([10., 33.3, 0.52]) # The true values of our parameter vector.
x = np.array([8., 43.5, 1.05]) # Initial guess of parameter vector for our solver


d = sinusoid(true_x, t) + noise # Our "observed" data, constructed from our true parameter values and the noise vector
m = sinusoid(x, t) # Our fitted function using the initial guess parameters.

# Plot Data and model
#plt.plot(t, d, label = 'Data')
#plt.plot(t, m, label = 'Model')
#plt.legend()
#plt.show()


print(x)
solved_x = levenberg_marquardt(d, t, x, sinusoid_residual, sinusoid_jacobian)
print(true_x)
print(solved_x)

plt.plot(t, d, label="Data")
plt.plot(t, sinusoid(solved_x, t), label="LM")
plt.plot(t, sinusoid(true_x, t), label="True")
plt.xlabel("t")
plt.legend()
plt.show()

