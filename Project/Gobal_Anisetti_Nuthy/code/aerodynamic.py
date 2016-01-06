# Author: Koorosh Gobal
# Python code for 3.3
# This function calculate airodynamic loads on the wedge
'''
Piston theory is an inviscid unsteady aerodynamic method used 
extensively in hypersonic aeroelasticity, which predicts a point function 
relationship between the local pressure on a lifting surface and the normal 
component of fluid velocity produced by the lifting surface motion. 
Here we use the third order expansion of "simple wave" expression for 
the pressure on a piston.

This code is based on the following paper:
@article{thuruthimattam2002aeroelasticity,
  title={Aeroelasticity of a generic hypersonic vehicle},
  author={Thuruthimattam, BJ and Friedmann, PP and McNamara, JJ and Powell, KG},
  journal={AIAA Paper},
  volume={1209},
  pages={2002},
  year={2002}
}
'''
# -----------------------------------
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pdb
# -----------------------------------
def calcLoad():
    gamma = 1.4 # For diatomic gas
    ainfty = 343.0 # Speed of sound (m/s)
    Pinfty = 10.0 # Free stream pressure
    V = 600.0 # Free stream velocity
    Zx = np.loadtxt('coord.txt')[:, 0]
    Zy = np.loadtxt('coord.txt')[:, 1]

    # Read the spatial part of the velocity
    dZdx = np.loadtxt('grad.txt')[:, 1]
    dZdt = np.loadtxt('coordDot.txt')[:, 1]

    # Calculate normal velocity
    vn = dZdt + V * dZdx

    # Calculate pressure
    P = Pinfty + Pinfty * (gamma * vn / ainfty + gamma * (gamma + 1)
                           * (vn / ainfty)**2 / 4 +
                           gamma * (gamma + 1) * (vn / ainfty)**3 / 12)

    # Calculate net force and moment around center of mass
    normalVec = np.loadtxt('normal.txt')
    dA = np.sqrt((Zx[1] - Zx[0])**2 + (Zy[1] - Zy[0])**2)
    Fx = np.multiply(P, normalVec[:, 0]) * dA
    Fy = np.multiply(P, normalVec[:, 1]) * dA
    Rx = -Zx
    Ry = -Zy

    Mz = np.zeros(len(Zx))
    for ni in range(0, len(Mz)):
        Mz[ni] = np.cross([Rx[ni], Ry[ni]], [Fx[ni], Fy[ni]])

    np.savetxt('P.txt', P, '%2.2f')
    np.savetxt('Fx.txt', Fx, '%2.2f')
    np.savetxt('Fy.txt', Fy, '%2.2f')
    Mz = -np.sum(Mz)
    Fy = np.sum(Fy)
    Fx = np.sum(Fx)

    return [Fx, Fy, Mz]
