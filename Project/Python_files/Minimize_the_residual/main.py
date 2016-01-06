import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp
from scipy.integrate import odeint
# # --------------------------------------------------------------------------------------------------------------------
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 52}
plt.rc('font', **font)
linewidth = 9.0
markersize = 20
# # # --------------------------------------------------------------------------------------------------------------------
# # \ddot{x} + x = sin(2*t)
# N = 99
# T = 2*np.pi
# t = np.linspace(0, T, N+1)
# t = t[0:-1]
# f = np.sin(2*t)
# F = np.fft.fft(f)
# Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
# x0 = np.ones(N)
#
# xAnalytical = - np.sin(2*t) / 3
# def residual(x):
#     X = np.fft.fft(x)
#     ddx = np.fft.ifft(np.multiply(-Omega**2, X))
#     R = ddx + x - f
#     R = np.sum(np.abs((R**2)))
#     return R
#
# res = minimize(residual, x0)
# print(residual(res.x))
# plt.figure()
# plt.plot(t, res.x,
#          t, xAnalytical)
# plt.show()

# # --------------------------------------------------------------------------------------------------------------------
# # \ddot{x} + \dot{x} + x = sin(2*t)
# N = 19
# T = 2*np.pi
# # t = np.linspace(0, 2*np.pi, N+1)
# t = np.linspace(0, T, N+1)
# t = t[0:-1]
# f = np.sin(2*t)
# F = np.fft.fft(f)
# Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
# x0 = np.ones(N)
# # x0 = f
# xAnalytical = -0.23077 * np.sin(2*t) -0.15385 * np.cos(2*t)
# residualHist = np.array([])
# def residual(x):
#     X = np.fft.fft(x)
#     ddx = np.fft.ifft(np.multiply(-Omega**2, X))
#     dx = np.fft.ifft(np.multiply(1j * Omega, X))
#     R = ddx + dx + x - f
#     # R = np.sum(np.abs(np.real(R)))
#     R = np.sum(np.abs((R**2)))
#     global residualHist
#     residualHist = np.append(residualHist, R)
#     np.savetxt('hist.txt', residualHist)
#     return R
#
# print(residual(xAnalytical))
# # res = minimize(residual, x0, options={'method':'SLSQP', 'maxiter':100000})
# res = minimize(residual, x0)
# print(residual(res.x))
# plt.figure(figsize=(30,15))
# plt.plot(t, res.x, 'k',
#          t, xAnalytical, 'r--o',
#          lw=linewidth, ms=markersize)
# plt.legend(['FFt', 'Analytical'])
# plt.xlabel('Time')
# plt.ylabel('Displacement')
# plt.savefig('1N19.eps', format='eps', dpi=1000, bbox_inches='tight')
# plt.show()
# print(res.jac)

# # --------------------------------------------------------------------------------------------------------------------
# # # \ddot{x} + \dot{x} + x - x**3 = sin(2*t)
# N = 199
# T = 4*2*np.pi
# t = np.linspace(0, T, N+1)
# t = t[0:-1]
# f = np.sin(2*t)
# F = np.fft.fft(f)
# Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
# x0 = np.zeros(N)
#
# def residual(x):
#     X = np.fft.fft(x)
#     ddx = np.fft.ifft(np.multiply(-Omega**2, X))
#     dx = np.fft.ifft(np.multiply(1j * Omega, X))
#     R = ddx + dx + x - x**3 - f
#     R = np.sum(np.abs((R**2)))
#     return R
#
# # res = minimize(residual, x0, options={'method':'SLSQP', 'maxiter':1000000})
# res = minimize(residual, x0)
# xSol = res.x
#
# # Numerical solution
# def RHS(X, t=0.0):
#     x1, x2 = X
#     x1dot = x2
#     x2dot = -x1 - x2 + x1**3 + np.sin(2*t)
#     return [x1dot, x2dot]
#
# ta = np.linspace(0.0, T, N)
# sol = odeint(RHS, [0, 0], ta)
# plt.figure()
# plt.plot(t, res.x, 'k',
#          ta, sol[:, 0], 'r')
# plt.legend(['Harmonic Balance', 'Time integration'])
# plt.xlabel('Time')
# plt.ylabel('Displacement')
# plt.show()

# # --------------------------------------------------------------------------------------------------------------------
# \ddot{x} + 2 * mu * \dot{x} + g / R * sin(x) - \alpha^2 * sin(x) * cos(x) = sin(2*t)
# Problem 2.45 on page 140 of Applied Nonlinear Dynamics: Analytical, Computational, and Experimental Methods (Nayfeh)
# Define system properties
# mu = 0.1
# g = 9.81
# R = 1.0
# alpha = 1
#
# N = 99
# T = 6*2*np.pi
# t = np.linspace(0, T, N+1)
# t = t[0:-1]
# f = np.sin(2*t)
# F = np.fft.fft(f)
# Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
# x0 = np.zeros(N)
#
# residualHist = np.array([])
# def residual(x):
#     X = np.fft.fft(x)
#     ddx = np.fft.ifft(np.multiply(-Omega**2, X))
#     dx = np.fft.ifft(np.multiply(1j * Omega, X))
#     Residual = ddx + 2 * mu * dx + g / R * np.sin(x) - alpha**2 * np.sin(x) * np.cos(x) - f
#     Residual = np.sum(np.abs((Residual**2)))
#     global residualHist
#     residualHist = np.append(residualHist, Residual)
#     np.savetxt('hist.txt', residualHist)
#     return Residual
#
# # res = minimize(residual, x0, options={'method':'SLSQP', 'maxiter':1000000})
# res = minimize(residual, x0)
# xSol = res.x

# # Numerical solution
# def RHS(X, t=0.0):
#     x1, x2 = X
#     x1dot = x2
#     x2dot = -2 * mu * x2 - g / R * np.sin(x1) + alpha**2 * np.sin(x1) * np.cos(x1) + np.sin(2*t)
#     return [x1dot, x2dot]
#
# ta = np.linspace(0.0, T, N)
# sol = odeint(RHS, [0, 0], ta)
# plt.figure(figsize=(30,15))
# plt.plot(t, res.x, 'k',
#          ta, sol[:, 0], 'r--',
#          lw=linewidth, ms=markersize)
# plt.legend(['FFt', 'Analytical'])
# plt.xlabel('Time')
# plt.ylabel('Displacement')
# plt.legend(['Harmonic Balance', 'Time integration'], loc='best')
# plt.xlabel('Time')
# plt.ylabel('Displacement')
# plt.savefig('2N199', format='eps', dpi=1000, bbox_inches='tight')
# plt.show()

# # # --------------------------------------------------------------------------------------------------------------------
# # \ddot{x} + x + epsilon * [2 * mu * \dpt{x} + alpha * x^3 + x * k * x * cos(omega * t)] = sin(2 * t)
# # Parametrically excited Duffing Oscillator
# # # Define system properties
epsilon = 1.0
mu = 1.0
alpha = 1.0
k = 1.0
omega = 2.0

N = 99
T = 2*2*np.pi
t = np.linspace(0, T, N+1)
t = t[0:-1]
Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
x0 = np.zeros(N)

residualHist = np.array([])
def residual(x):
    X = np.fft.fft(x)
    dx = np.fft.ifft(np.multiply(1j * Omega, X))
    ddx = np.fft.ifft(np.multiply(-Omega**2, X))
    Residual = ddx + x + epsilon * (2 * mu * dx + alpha * x**3 + 2 * k * x * np.cos(omega * t)) - np.sin(2 * t)
    Residual = np.sum(np.abs((Residual**2)))
    global residualHist
    residualHist = np.append(residualHist, Residual)
    np.savetxt('hist.txt', residualHist)
    return Residual

# res = minimize(residual, x0, options={'method':'SLSQP', 'maxiter':1000000})
res = minimize(residual, x0)
xSol = res.x

# Numerical solution
# def RHS(X, t=0.0):
#     x1, x2 = X
#     x1dot = x2
#     x2dot = -x1 - epsilon * (2 * mu * x2 + alpha * x1**3 + 2 * k * x1 * np.cos(omega * t)) + np.sin(2 * t)
#     return [x1dot, x2dot]
#
# ta = np.linspace(0.0, T, N)
# sol = odeint(RHS, [0, 0], ta)
# plt.figure(figsize=(30,15))
# plt.plot(t, res.x, 'k',
#          ta, sol[:, 0], 'r--',
#          lw=linewidth, ms=markersize)
# plt.legend(['Harmonic Balance', 'Time integration'], loc='best')
# plt.xlabel('Time')
# plt.ylabel('Displacement')
# plt.savefig('3N20.eps', format='eps', dpi=1000, bbox_inches='tight')
# plt.show()

# # # --------------------------------------------------------------------------------------------------------------------
# # # \ddot{x} + x = sin(2*t)
# N = 99
# T = 3*np.pi
# t = np.linspace(0, T, N+1)
# t = t[0:-1]
# f = 1 * np.cos(2 * t)
# F = np.fft.fft(f)
# Omega = np.fft.fftfreq(N, T/(2*np.pi*N))
# x0 = np.ones(N)
#
# xAnalytical = - np.sin(2*t) / 3
# def residual(x):
#     X = np.fft.fft(x)
#     ddx = np.fft.ifft(np.multiply(-Omega**2, X))
#     dx = np.fft.ifft(np.multiply(1j*Omega, X))
#     R = ddx - 1.0 * np.multiply(1 - x**2, dx) + x - f
#     R = np.sum(np.abs((R**2)))
#     return R
#
# res = minimize(residual, x0)
# print(residual(res.x))
# print(res.jac)
# plt.figure()
# plt.plot(t, res.x)
# plt.show()