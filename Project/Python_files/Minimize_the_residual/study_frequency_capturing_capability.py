__author__ = 'koorosh'
import numpy as np
# # Study for which frequencies can be captured
T = 4 * np.pi
N = 10
t = np.linspace(0, T, N+1)
t = t[:-1]
n = np.fft.fftfreq(N, 1/N)
omega0 = 2*np.pi / T
omega = omega0 * n

a = 3.0
x = np.sin(a*t)

print(np.fft.fft(x))
print(omega)

np.savetxt('X.txt', np.fft.fft(x) / N, fmt='%4.4f')
np.savetxt('omega.txt', omega, fmt='%4.4f')

# a = 0.5
# x = np.sin(a*t)
# dxdt = a*np.cos(a*t)
#
# X = np.fft.fft(x)
# DxDt = np.fft.ifft(1j * np.multiply(omega, X))
#
# print(n)
# print(np.max(np.abs(np.real(DxDt - dxdt))))