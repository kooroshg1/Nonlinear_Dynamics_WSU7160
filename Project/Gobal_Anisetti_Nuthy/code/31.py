# Author: Koorosh Gobal
# Python code for 3.1
# -----------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
# -----------------------------------
# Governing Equation: \ddot{x} + \dot{x} + x = sin(2t)
N = 19
T = 2*np.pi
t = np.linspace(0, T, N+1)
t = t[0:-1]
f = np.sin(2*t)
Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
x0 = np.ones(N)

xAnalytical = -0.23077 * np.sin(2*t) -0.15385 * np.cos(2*t)
def residual(x):
    X = np.fft.fft(x)
    ddx = np.fft.ifft(np.multiply(-Omega**2, X))
    dx = np.fft.ifft(np.multiply(1j * Omega, X))
    R = ddx + dx + x - f
    R = np.sum(np.abs(np.real(R)))
    R = np.sum(np.abs((R**2)))
    return R
#
res = minimize(residual, x0, method='SLSQP')
print('Residual at the solution: ')
print(residual(res.x))
print('Jacobian of residual at the solution: ')
print(res.jac)
plt.figure()
plt.plot(t, res.x, 'k',
         t, xAnalytical, 'r--o')
plt.legend(['FFt', 'Analytical'])
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.show()
