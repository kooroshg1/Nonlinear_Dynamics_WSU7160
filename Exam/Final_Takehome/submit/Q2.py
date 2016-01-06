# Python code for Q2 of ME7160
__author__ = 'koorosh gobal'
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
# # ----------------------------------------
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)
linewidth = 5.0
markersize = 15
skip = 10
# # ----------------------------------------
# Harmonic Balance (HB) method
N = 19 # Number of harmonics for HB
T = 1 * 2 * np.pi # Time interval
t = np.linspace(0, T, N+1)
t = t[0:-1]
f = np.sin(3*t) # Forcing function
Omega = 2*np.pi / T * np.fft.fftfreq(N, 1/N) # \omega
x0 = np.ones(N) # Initial guess for the solution
alpha = 12 # \alpha for this problem

# defining residual function of HB method
def residual(x):
    X = np.fft.fft(x)
    dx = np.fft.ifft(np.multiply(1j * Omega, X))
    ddx = np.fft.ifft(np.multiply(-Omega**2, X))
    Residual = ddx + (0.01 + alpha / 100) * dx + x + alpha / 10 * x**3 - f
    Residual = np.sum(np.abs((Residual**2)))
    return Residual
    
res = minimize(residual, x0, method='SLSQP') # Minimizing the residual
xSol = res.x # Solution in time domain

# Converting result to sine and cosines
n = 3 # Number of harmonics used
X = np.fft.fft(xSol)
X = np.append(X[:n+1], X[-n:]) / N
OMEGA = np.append(Omega[:n+1], Omega[-n:]).reshape(1, -1)
tReconst = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
OMEGAt = tReconst.dot(OMEGA).T
xReconst = X[0] # Reconstracuted results
for i in range(1, OMEGA.shape[1]):
    xReconst = xReconst + np.real(X[i]) * np.cos(OMEGAt[i, :]) - \
                          np.imag(X[i]) * np.sin(OMEGAt[i, :])
np.savetxt('X.txt', X*2)
np.savetxt('OMEGA.txt', OMEGA.T)

# Numerical solution using odeint
def RHS(X, t=0.0):
    x1, x2 = X
    x1dot = x2
    x2dot = -(0.01 + alpha / 100) * x2 - x1 - alpha / 10 * x1**3 \
            + np.sin(3*t)
    return [x1dot, x2dot]

ta = np.linspace(0.0, 19*2*np.pi + t[-1], 5000)
sol = odeint(RHS, [0, 0], ta)

# Ploting the comparing the results
# We have to shift the HB result to region where the transient
# Solution is died out and the particular solution is the dominent
# one. This is what '19*2*np.pi' does.
plt.figure()
plt.plot(19*2*np.pi + t, res.x, 'k',
         ta[0:-1:skip], sol[0:-1:skip, 0], 'wo',
         19*2*np.pi + tReconst, xReconst, 'k--',
         lw=linewidth, ms=markersize, mew=linewidth)
plt.legend(['HB', 'Time integration', 'HB with 3 Harmonics'], loc='best')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.xlim([19*2*np.pi + t[0], 19*2*np.pi + t[-1]])
plt.title(r'$\alpha = 12$')
plt.show()
