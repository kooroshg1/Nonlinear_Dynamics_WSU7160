__author__ = 'koorosh'
import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 52}

def potential(u, a, Lambda):
    return 0.5 * u**2 + Lambda * np.log(np.abs(a - u))
def equilibrium_points(a, Lambda):
    equi1 = (a + (a**2.0 - 4.0*Lambda)**0.5) / 2.0
    equi2 = (a - (a**2.0 - 4.0*Lambda)**0.5) / 2.0
    if np.imag(equi1) != 0:
        equi1 = np.NAN
    if np.imag(equi2) != 0:
        equi2 = np.NAN
    return [equi1, equi2]
def mechanical(u, udot, a, Lambda):
    return 0.5*udot**2 + potential(u, a, Lambda)

a = [1.0, 2.0, 3.0, 4.0, 5.0]
Lambda = [0.0, 0.2, 1.0, 4.0, 7.0]

u= np.linspace(-10.0, 10.0, 1000)
udot= np.linspace(-10.0, 10.0, 1000)
U, Udot = np.meshgrid(u, udot)
for n in range(0, len(a)):
    equi1, equi2 = equilibrium_points(a[n], Lambda[n])
    print(equi1)
    print(equi2)
    plt.figure(1)
    plt.plot(u, potential(u, a[n], Lambda[n]), 'k-',
             equi1, potential(equi1, a[n], Lambda[n]), 'wo',
             equi2, potential(equi2, a[n], Lambda[n]), 'wo',
             ms=10.0, mew=2.0, linewidth=2.0)
    plt.xlabel('$u_1$')
    plt.ylabel('$F(u_1)$')
    plt.rc('font', **font)
    filename = str('mechanical_a' + str(int(a[n]*10.0)) + '_lambda' + str(int(Lambda[n]*10.0)) + '.eps')

    # Plots the contour of mechanical energy
    plt.figure(figsize=(16, 12))
    plt.contour(U, Udot, mechanical(U, Udot, a[n], Lambda[n]), 50)
    plt.plot([equi1, equi2], [0.0, 0.0], 'wo', ms=20.0, mew=2.0)
    plt.xlabel('$u$')
    plt.ylabel('$\dot{u}$')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig(filename, format='eps')

#plt.show()