__author__ = 'koorosh gobal'
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
# # --------------------------------------------------------------------------------------------------------------------
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)
linewidth = 9.0
markersize = 20
# # --------------------------------------------------------------------------------------------------------------------
# \ddot{x} + \dot{x} + x = sin(2*t)
T = 8 * np.pi
# T = 10
N = 3
t = np.linspace(0, T, N+1)
t = t[:-1]
n = np.fft.fftfreq(N, 1/N)
omega0 = 2*np.pi / T
omega = omega0 * n

x0 = np.ones(N)
# x0 = np.sin(2*t)
x0 = np.concatenate((x0, [T]))
def residual(x):
    T = x[-1]
    x = x[:-1]
    omega0 = 2*np.pi / T
    omega = omega0 * n
    t = np.linspace(0, T, N+1)
    t = t[0:-1]
    f = np.sin(2*t)

    # # Check derivatives
    # a = 2
    # x_ = np.sin(a*t)
    # dxdt_ = a*np.cos(a*t)
    #
    # X_ = np.fft.fft(x_)
    # DxDt_ = np.fft.ifft(1j * np.multiply(omega, X_))
    #
    # print(np.max(np.abs(np.real(DxDt_ - dxdt_))))

    X = np.fft.fft(x)
    ddx = np.fft.ifft(np.multiply(-omega**2, X))
    dx = np.fft.ifft(np.multiply(1j * omega, X))
    R = ddx + x - f
    # R = np.sum(np.abs(np.real(R)))
    R = np.sum(np.abs((R**2)))
    return R

# res = minimize(residual, x0, method='SLSQP', options={'maxiter':1000000})
res = minimize(residual, x0)
# print(res.x)
# print(res.jac)
# print(residual(res.x))

# # # Verification of results
# xFFTsol = res.x[:-1]
# T = res.x[-1]
# t = np.linspace(0, T, N+1)
# t = t[0:-1]
# f = np.sin(2*t)
# # xAnalytical = -0.23077 * np.sin(2*t) -0.15385 * np.cos(2*t)
# xAnalytical = -1/3 * np.sin(2*t)
# # # plt.figure(figsize=(30,15))
# plt.figure()
# plt.plot(t, xFFTsol, 'k',
#          t, xAnalytical, 'r--o',
#          lw=linewidth, ms=markersize)
# plt.legend(['FFt', 'Analytical'])
# plt.xlabel('Time')
# plt.ylabel('Displacement')
# plt.savefig('1N19.eps', format='eps', dpi=1000, bbox_inches='tight')
# plt.show()

# # # --------------------------------------------------------------------------------------------------------------------
# # \ddot{x} + x + epsilon * [2 * mu * \dpt{x} + alpha * x^3 + x * k * x * cos(omega * t)] = sin(2 * t)
# # Parametrically excited Duffing Oscillator
# # # Define system properties
# epsilon = 1.0
# mu = 1.0
# alpha = 1.0
# k = 1.0
# omega = 2.0
#
# T = 2*np.pi
# N = 9
# t = np.linspace(0, T, N+1)
# t = t[:-1]
# n = np.fft.fftfreq(N, 1/N)
# omega0 = 2*np.pi / T
# Omega = omega0 * n
#
# x0 = np.zeros(N)
# x0 = np.concatenate((x0, [T]))
#
# def residual(x):
#     T = x[-1]
#     x = x[:-1]
#     omega0 = 2*np.pi / T
#     Omega = omega0 * n
#     t = np.linspace(0, T, N+1)
#     t = t[0:-1]
#
#     X = np.fft.fft(x)
#     dx = np.fft.ifft(np.multiply(1j * Omega, X))
#     ddx = np.fft.ifft(np.multiply(-Omega**2, X))
#     Residual = ddx + x + epsilon * (2 * mu * dx + alpha * x**3 + 2 * k * x * np.cos(omega * t)) - np.sin(2 * t)
#     Residual = np.sum(np.abs((Residual**2)))
#     return Residual
#
# # res = minimize(residual, x0, options={'method':'SLSQP', 'maxiter':1000000})
# res = minimize(residual, x0)
# xSol = res.x[:-1]
# T = res.x[-1]
#
#
# # Numerical solution
# def RHS(X, t=0.0):
#     x1, x2 = X
#     x1dot = x2
#     x2dot = -x1 - epsilon * (2 * mu * x2 + alpha * x1**3 + 2 * k * x1 * np.cos(omega * t)) + np.sin(2 * t)
#     return [x1dot, x2dot]
#
# ta = np.linspace(0.0, T, 1000)
# sol = odeint(RHS, [0, 0], ta)
# # plt.figure(figsize=(30,15))
# plt.figure()
# plt.plot(t, xSol, 'k',
#          ta, sol[:, 0], 'r--',
#          lw=linewidth, ms=markersize)
# plt.legend(['Harmonic Balance', 'Time integration'], loc='best')
# plt.xlabel('Time')
# plt.ylabel('Displacement')
# # plt.savefig('3N20.eps', format='eps', dpi=1000, bbox_inches='tight')
# plt.show()
#
