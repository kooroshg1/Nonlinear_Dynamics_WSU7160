# Author: Koorosh Gobal
# Python code for 3.2
# -----------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
# -----------------------------------
# \ddot{x} + 2 * mu * \dot{x} + g / R * sin(x) - \alpha^2 * sin(x) * cos(x) = sin(2*t)
# Problem 2.45 on page 140 of Applied Nonlinear Dynamics: Analytical, Computational, and Experimental Methods (Nayfeh)
# Define system properties
mu = 0.1
g = 9.81
R = 1.0
alpha = 1

N = 99
T = 6*2*np.pi
t = np.linspace(0, T, N+1)
t = t[0:-1]
Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
x0 = np.zeros(N)

# Harmonic Balance method
def residual(x):
    X = np.fft.fft(x)
    ddx = np.fft.ifft(np.multiply(-Omega**2, X))
    dx = np.fft.ifft(np.multiply(1j * Omega, X))
    Residual = ddx + 2 * mu * dx + g / R * np.sin(x) - alpha**2 * np.sin(x) * np.cos(x) - np.sin(2*t)
    Residual = np.sum(np.abs((Residual**2)))
    return Residual
#
res = minimize(residual, x0, method='SLSQP')
xSol = res.x

# Numerical solution
def RHS(X, t=0.0):
    x1, x2 = X
    x1dot = x2
    x2dot = -2 * mu * x2 - g / R * np.sin(x1) + alpha**2 * np.sin(x1) * np.cos(x1) + np.sin(2*t)
    return [x1dot, x2dot]
#
ta = np.linspace(0.0, T, N)
sol = odeint(RHS, [0, 0], ta)

plt.figure()
plt.plot(t, res.x, 'k',
         ta, sol[:, 0], 'r--')
plt.legend(['FFt', 'Analytical'])
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend(['Harmonic Balance', 'Time integration'], loc='best')
plt.show()
