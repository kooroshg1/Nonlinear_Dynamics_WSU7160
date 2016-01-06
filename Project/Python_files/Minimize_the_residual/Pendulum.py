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
m1 = 2
m2 = 1
l1 = 1
l2 = 2
g = 32.2
c1 = 1
c2 = .5

N = 99
T = 2*np.pi
t = np.linspace(0, T, N+1)
t = t[0:-1]
p = np.sin(0.2*t)
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
    a = (m1+m2)*l1
    b = m2*l2*np.cos(x1-x2)
    c = m2*l1*np.cos(x1-x2)
    d = m2*l2
    e = -m2*l2*dx2**2*np.sin(x1-x2)-g*(m1+m2)*np.sin(x1)+(c1+c2)*(l1**2)*dx1+c2*l1*l2*dx2
    f = m2*l1*dx1**2*np.sin(x1-x2)-m2*g*np.sin(x2)+p+c2*dx2+c2*l1*l2*dx1+c2*(l2**2)*dx2
    R1 = a*ddx1+b*ddx2-e
    R2 = c*ddx2+d*ddx1-f
    Residual = R1**2 + R2**2
    Residual = np.sum(np.abs((Residual)))
    return Residual

res = minimize(residual, x0)
print(residual(res.x))
xSol = res.x
xSol1 = xSol[:N]
xSol2 = xSol[N:]

# Numerical solution
def RHS(X, t=0.0):
     x11, x12, x21, x22 = X
     x11dot = x12
     x21dot = x22
     a = (m1+m2)*l1
     b = m2*l2*np.cos(x11-x21)
     c = m2*l1*np.cos(x11-x21)
     d = m2*l2
     e = -m2*l2*x22**2*np.sin(x11-x21)-g*(m1+m2)*np.sin(x11)+(c1+c2)*l1**2*x12+c2*l1*l2*x22
     f = m2*l1*x12**2*np.sin(x11-x21)-m2*g*np.sin(x21)+np.sin(0.2*t)+c2*l1*l2*x12+c2*l2**2*x22
     x12dot = (e*d-b*f)/(a*d-c*b)
     x22dot = (a*f-c*e)/(a*d-c*b)
     return [x11dot, x12dot, x21dot, x22dot]
#
ta = np.linspace(0.0, T, 20*N)
sol = odeint(RHS, [0, 0, 0, 0], ta)
print(sol)

plt.figure()
plt.plot(t, xSol1, 'k',
         ta, sol[:, 0], 'r--',
         lw=linewidth, ms=markersize)
plt.legend(['Harmonic Balance', 'Time integration'], loc='best')
plt.title('theta1')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.show()
#
plt.figure()
plt.plot(t, xSol2, 'k',
         ta, sol[:, 2], 'r--',
         lw=linewidth, ms=markersize)
plt.legend(['Harmonic Balance', 'Time integration'], loc='best')
plt.title('theta2')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.show()