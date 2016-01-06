__author__ = 'koorosh'
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp
from scipy.integrate import odeint
# # -------------------------------------------------------------------------------------------------------------------
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)
linewidth = 4.0
markersize = 10
# # -------------------------------------------------------------------------------------------------------------------
# # 2DOF Mass-Spring-Damper System
# # Define system properties
m1 = 1
m2 = 1
k1 = 1
k2 = 1
c1 = 1

N = 99
T = 5*2*np.pi
t = np.linspace(0, T, N+1)
t = t[0:-1]
f = np.sin(2*t)
F = np.fft.fft(f)
Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
x0 = np.zeros(N*2)

def residual(x):
    x1 = x[:N]
    x2 = x[N:]

    X1 = np.fft.fft(x1)
    X2 = np.fft.fft(x2)
    dx1 = np.fft.ifft(np.multiply(1j * Omega, X1))
    dx2 = np.fft.ifft(np.multiply(1j * Omega, X2))
    ddx1 = np.fft.ifft(np.multiply(-Omega**2, X1))
    ddx2 = np.fft.ifft(np.multiply(-Omega**2, X2))

    Residual1 = m1 * ddx1 + c1 * dx1 + (k1 + k2) * x1 - k2 * x2
    Residual2 = m2 * ddx2 - k2 * x1 + k2 * x2 - f
    Residual = Residual1**2 + Residual2**2
    Residual = np.sum(np.abs((Residual)))
    return Residual

# # res = minimize(residual, x0, options={'method':'SLSQP', 'maxiter':1000000})
res = minimize(residual, x0)
print(residual(res.x))
xSol = res.x
xSol1 = xSol[:N]
xSol2 = xSol[N:]

# Numerical solution
def RHS(X, t=0.0):
    x11, x12, x21, x22 = X
    x11dot = x12
    x12dot = -(k1 + k2) / m1 * x11 - c1 / m1 * x12 + k2 / m1 * x21
    x21dot = x22
    x22dot = k2 / m2 * x11 - k2 / m1 * x21 + np.sin(2*t) / m1
    return [x11dot, x12dot, x21dot, x22dot]

ta = np.linspace(0.0, T, 20*N)
sol = odeint(RHS, [0, 0, 0, 0], ta)
# plt.figure(figsize=(30,15))
plt.figure()
plt.plot(t, xSol1, 'k',
         ta, sol[:, 0], 'r--',
         lw=linewidth, ms=markersize)
plt.legend(['FFt', 'Analytical'])
plt.legend(['Harmonic Balance', 'Time integration'], loc='best')
plt.title('x1')
plt.xlabel('Time')
plt.ylabel('Displacement')
# plt.savefig('2N199', format='eps', dpi=1000, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(t, xSol2, 'k',
         ta, sol[:, 2], 'r--',
         lw=linewidth, ms=markersize)
plt.legend(['FFt', 'Analytical'])
plt.legend(['Harmonic Balance', 'Time integration'], loc='best')
plt.title('x2')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.show()