import numpy as np
import matplotlib.pyplot as plt


"""
    We consider a 2-element interferometer projecting a baseline (u, v=0, w=0) associated
    with a set of fringe along the m-axis (for simplification).
    The sky is only composed of a single extended source represented by a disk of unit brightness.
    The inteferometer observe the source at the phase center.
    Only source in the sky.
    Effect of antenna patter is negligible.
    w = 0
    Visibility function becomes : 
    V = Integral_over_disk(exp(-2iPIul)dl)
"""

from matplotlib.patches import Circle

def plotfringe(u=4, v=0, rad=0.2):
    
    # angular Radius of the object in l,m coordinate
    radius = rad
    # preparing (l,m,n) space
    Npointsl = 1001
    ll=np.linspace(-1.,1.,Npointsl)
    l,m = np.meshgrid(ll, ll)

    # Projected baseline length on the u-axis
    #u = u
    
    # Generate fringe pattern
    tabcos = np.real(np.exp(-2j*np.pi*(u*l+v*m)))

    # Plotting the fringe pattern of the source
    pxrad = radius*Npointsl/2
    circle = Circle((500,500), pxrad, color='r', alpha=0.5, fill=True)

    fig, ax = plt.subplots(figsize=(6,6))
    im = plt.imshow(np.abs(tabcos), interpolation=None, cmap='winter')
    ax.add_patch(circle)

    # Compute the absolute value of the integral of the fringe over the source
    w = np.where (np.sqrt(l**2+m**2) <= radius)
    integral = np.sum(tabcos[w])
    print("Integral =" +str(integral))

    plt.show()
    

def plot_integral(umax=15):
    radius = 0.2

    Npointsl = 1001
    Npointsu = 500 # nb of u to compute

    
    ll = np.linspace(-1., 1., Npointsl)
    l, m = np.meshgrid(ll, ll)
    u = np.arange(start=0, stop=umax, step=umax/Npointsu)
    
    w = np.where(np.sqrt(l**2+m**2) <= radius)

    integral = np.array([])
    print(integral)
    for du in u:
        tabcos = np.real(np.exp(-2j*np.pi*du*l))
        integral = np.append(integral, np.abs(np.sum(tabcos[w])))

    normintegral = integral/np.max(integral)
    plt.plot(u, normintegral)
    plt.show()


#plotfringe(u=4, v=0, rad=0.2)
plot_integral()
