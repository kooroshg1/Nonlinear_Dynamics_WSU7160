import numpy as np
import matplotlib.pyplot as plt

# # Matching derivative of sin(x) using FFTs
# N = 4
# t = np.linspace(0, 2*np.pi, N+1)
# t = t[0:-1]
# x = np.sin(t)
# X = np.fft.fft(x)
# omega = np.fft.fftfreq(len(X), 1/N)
# print(omega * 1j * np.round(X))
# print(np.round(np.fft.fft(np.cos(t))))

# # \ddot{x} + x = \sin(2t)
N = 9
t = np.linspace(0, 2*np.pi, N+1)
t = t[0:-1]
f = np.sin(2*t)
F = np.fft.fft(f)
Omega = np.fft.fftfreq(N, 1/N) + 0.00001
X = np.divide(F,1 - Omega**2)
x = np.fft.ifft(X)

xAnalytical = - np.sin(2*t) / 3

plt.figure()
plt.subplot(211)
plt.plot(t, x, 'k',
         t, xAnalytical, 'ro')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend(['FFT', 'Analytical'])
plt.subplot(212)
plt.plot(t, 100 * np.divide(np.abs(x - xAnalytical),np.abs(xAnalytical)), 'k')
plt.xlabel('Time')
plt.ylabel('% Error')
# plt.show()

# # \ddot{x} + \dot{x} + x = \sin(2t)
N = 99
t = np.linspace(0, 2*np.pi, N+1)
t = t[0:-1]
f = np.sin(2*t)
F = np.fft.fft(f)
Omega = np.fft.fftfreq(N, 1/N) + 0.00001
X = np.divide(F,1 + 1j * Omega - Omega**2)
x = np.fft.ifft(X)

xAnalytical = -0.23077 * np.sin(2*t) -0.15385 * np.cos(2*t)

plt.figure()
plt.subplot(211)
plt.plot(t, x, 'k',
         t, xAnalytical, 'ro')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend(['FFT', 'Analytical'])
plt.subplot(212)
plt.plot(t, 100 * np.divide(np.abs(x - xAnalytical),np.abs(xAnalytical)), 'k')
plt.xlabel('Time')
plt.ylabel('% Error')
plt.show()

