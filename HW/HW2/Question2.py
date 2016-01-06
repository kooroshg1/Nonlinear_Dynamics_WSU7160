__author__ = 'koorosh'
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
font = {'size'   : 62}

plt.rc('font', **font)

# 2A) \ddot{x} + x + x^3 = 0
#'''
def RHS2A(X, t=0.0):
    x1, x2 = X
    x1dot = x2
    x2dot = -x1 - x1**3
    return [x1dot, x2dot]


x1 = np.linspace(-5.0, 5.0, 50)
x2 = np.linspace(-5.0, 5.0, 50)
X1, X2 = np.meshgrid(x1, x2)
X1dot = np.zeros(X1.shape)
X2dot = np.zeros(X2.shape)
normalize = True
for ni in range(0, len(x1)):
    for nj in range(0, len(x2)):
        X1dot[ni, nj], X2dot[ni, nj] = RHS2A([X1[ni, nj], X2[ni, nj]])
        if normalize:
            X1dot[ni, nj] = X1dot[ni, nj] / (X1dot[ni, nj]**2.0 + X2dot[ni, nj]**2.0)**0.5
            X2dot[ni, nj] = X2dot[ni, nj] / (X1dot[ni, nj]**2.0 + X2dot[ni, nj]**2.0)**0.5

plt.figure(figsize=[20, 20])
plt.quiver(X1, X2, X1dot, X2dot)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([x1[0], x1[-1]])
plt.ylim([x2[0], x2[-1]])

x10 = [0.5, 1.5, 2.0]
x20 = [0.0, -1.0, 1.5]
t = np.linspace(0.0, 10.0, 1000)
t = np.linspace(0.0, 10.0, 1000)
for nInit in range(0, len(x10)):
    sol = odeint(RHS2A, [x10[nInit], x20[nInit]], t)
    for nSol in range(0, len(sol)):
        if sol[nSol, 0] > 100.0:
            break
        elif sol[nSol, 1] > 100:
            break
        elif sol[nSol, 0] < -100:
            break
        elif sol[nSol, 1] < -100:
            break
    plt.plot(sol[0:nSol, 0], sol[0:nSol, 1], lw=4.0)
    plt.plot(sol[0, 0], sol[0, 1], 's', ms=20)
    plt.plot(sol[nSol, 0], sol[nSol, 1], 'o', ms=20)

plt.savefig('/home/koorosh/Desktop/Nonlinear_Dynamics/HW/HW2/Report/figure/2A.eps', bbox_inches='tight')
plt.show()
#'''
# -----------------------------------------------------------------
# 2B) \ddot{x} + x - x^3 = 0
#'''
def RHS2A(X, t=0.0):
    x1, x2 = X
    x1dot = x2
    x2dot = -x1 + x1**3.0
    return [x1dot, x2dot]


x1 = np.linspace(-5.0, 5.0, 50)
x2 = np.linspace(-5.0, 5.0, 50)
X1, X2 = np.meshgrid(x1, x2)
X1dot = np.zeros(X1.shape)
X2dot = np.zeros(X2.shape)
normalize = True
for ni in range(0, len(x1)):
    for nj in range(0, len(x2)):
        X1dot[ni, nj], X2dot[ni, nj] = RHS2A([X1[ni, nj], X2[ni, nj]])
        if normalize:
            X1dot[ni, nj] = X1dot[ni, nj] / (X1dot[ni, nj]**2.0 + X2dot[ni, nj]**2.0)**0.5
            X2dot[ni, nj] = X2dot[ni, nj] / (X1dot[ni, nj]**2.0 + X2dot[ni, nj]**2.0)**0.5

plt.figure(figsize=[20, 20])
plt.quiver(X1, X2, X1dot, X2dot)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([x1[0], x1[-1]])
plt.ylim([x2[0], x2[-1]])

x10 = [0.5, 1.5, 2.0]
x20 = [0.0, -1.0, 1.5]
t = np.linspace(0.0, 10.0, 1000)
for nInit in range(0, len(x10)):
    sol = odeint(RHS2A, [x10[nInit], x20[nInit]], t)
    for nSol in range(0, len(sol)):
        if sol[nSol, 0] > 100.0:
            break
        elif sol[nSol, 1] > 100:
            break
        elif sol[nSol, 0] < -100:
            break
        elif sol[nSol, 1] < -100:
            break
    plt.plot(sol[0:nSol, 0], sol[0:nSol, 1], lw=4.0)
    plt.plot(sol[0, 0], sol[0, 1], 's', ms=20)
    plt.plot(sol[nSol, 0], sol[nSol, 1], 'o', ms=20)

plt.savefig('/home/koorosh/Desktop/Nonlinear_Dynamics/HW/HW2/Report/figure/2B.eps', bbox_inches='tight')
plt.show()
#'''
# -----------------------------------------------------------------
# 2C) \ddot{x} - x + x^3 = 0
#'''
def RHS2A(X, t=0.0):
    x1, x2 = X
    x1dot = x2
    x2dot = x1 - x1**3.0
    return [x1dot, x2dot]


x1 = np.linspace(-5.0, 5.0, 50)
x2 = np.linspace(-5.0, 5.0, 50)
X1, X2 = np.meshgrid(x1, x2)
X1dot = np.zeros(X1.shape)
X2dot = np.zeros(X2.shape)
normalize = True
for ni in range(0, len(x1)):
    for nj in range(0, len(x2)):
        X1dot[ni, nj], X2dot[ni, nj] = RHS2A([X1[ni, nj], X2[ni, nj]])
        if normalize:
            X1dot[ni, nj] = X1dot[ni, nj] / (X1dot[ni, nj]**2.0 + X2dot[ni, nj]**2.0)**0.5
            X2dot[ni, nj] = X2dot[ni, nj] / (X1dot[ni, nj]**2.0 + X2dot[ni, nj]**2.0)**0.5

plt.figure(figsize=[20, 20])
plt.quiver(X1, X2, X1dot, X2dot)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([x1[0], x1[-1]])
plt.ylim([x2[0], x2[-1]])

x10 = [0.5, 1.5, 2.0]
x20 = [0.0, -1.0, 1.5]
t = np.linspace(0.0, 10.0, 100)
for nInit in range(0, len(x10)):
    sol = odeint(RHS2A, [x10[nInit], x20[nInit]], t)
    for nSol in range(0, len(sol)):
        if sol[nSol, 0] > 100.0:
            break
        elif sol[nSol, 1] > 100:
            break
        elif sol[nSol, 0] < -100:
            break
        elif sol[nSol, 1] < -100:
            break
    plt.plot(sol[0:nSol, 0], sol[0:nSol, 1], lw=4.0)
    plt.plot(sol[0, 0], sol[0, 1], 's', ms=20)
    plt.plot(sol[nSol, 0], sol[nSol, 1], 'o', ms=20)

plt.savefig('/home/koorosh/Desktop/Nonlinear_Dynamics/HW/HW2/Report/figure/2C.eps', bbox_inches='tight')
plt.show()
#'''
# -----------------------------------------------------------------
# 2D) \ddot{x} - x - x^3 = 0
#'''
def RHS2A(X, t=0.0):
    x1, x2 = X
    x1dot = x2
    x2dot = x1 + x1**3.0
    return [x1dot, x2dot]

x1 = np.linspace(-5.0, 5.0, 50)
x2 = np.linspace(-5.0, 5.0, 50)
X1, X2 = np.meshgrid(x1, x2)
X1dot = np.zeros(X1.shape)
X2dot = np.zeros(X2.shape)
normalize = True
for ni in range(0, len(x1)):
    for nj in range(0, len(x2)):
        X1dot[ni, nj], X2dot[ni, nj] = RHS2A([X1[ni, nj], X2[ni, nj]])
        if normalize:
            X1dot[ni, nj] = X1dot[ni, nj] / (X1dot[ni, nj]**2.0 + X2dot[ni, nj]**2.0)**0.5
            X2dot[ni, nj] = X2dot[ni, nj] / (X1dot[ni, nj]**2.0 + X2dot[ni, nj]**2.0)**0.5

plt.figure(figsize=[20, 20])
plt.quiver(X1, X2, X1dot, X2dot)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([x1[0], x1[-1]])
plt.ylim([x2[0], x2[-1]])

x10 = [0.5, 1.5, 2.0]
x20 = [0.0, -1.0, 1.5]
t = np.linspace(0.0, 10.0, 1000)
for nInit in range(0, len(x10)):
    sol = odeint(RHS2A, [x10[nInit], x20[nInit]], t)
    for nSol in range(0, len(sol)):
        if sol[nSol, 0] > 100.0:
            break
        elif sol[nSol, 1] > 100:
            break
        elif sol[nSol, 0] < -100:
            break
        elif sol[nSol, 1] < -100:
            break
    plt.plot(sol[0:nSol, 0], sol[0:nSol, 1], lw=4.0)
    plt.plot(sol[0, 0], sol[0, 1], 's', ms=20)
    plt.plot(sol[nSol, 0], sol[nSol, 1], 'o', ms=20)

plt.savefig('/home/koorosh/Desktop/Nonlinear_Dynamics/HW/HW2/Report/figure/2D.eps', bbox_inches='tight')
plt.show()
#'''
# -----------------------------------------------------------------
# 2E) \ddot{x} + x^3 = 0
#'''
def RHS2A(X, t=0.0):
    x1, x2 = X
    x1dot = x2
    x2dot = -x1**3.0
    return [x1dot, x2dot]

x1 = np.linspace(-5.0, 5.0, 50)
x2 = np.linspace(-5.0, 5.0, 50)
X1, X2 = np.meshgrid(x1, x2)
X1dot = np.zeros(X1.shape)
X2dot = np.zeros(X2.shape)
normalize = True
for ni in range(0, len(x1)):
    for nj in range(0, len(x2)):
        X1dot[ni, nj], X2dot[ni, nj] = RHS2A([X1[ni, nj], X2[ni, nj]])
        if normalize:
            X1dot[ni, nj] = X1dot[ni, nj] / (X1dot[ni, nj]**2.0 + X2dot[ni, nj]**2.0)**0.5
            X2dot[ni, nj] = X2dot[ni, nj] / (X1dot[ni, nj]**2.0 + X2dot[ni, nj]**2.0)**0.5

plt.figure(figsize=[20, 20])
plt.quiver(X1, X2, X1dot, X2dot)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([x1[0], x1[-1]])
plt.ylim([x2[0], x2[-1]])

x10 = [0.5, 1.5, 2.0]
x20 = [0.0, -1.0, 1.5]
t = np.linspace(0.0, 50.0, 1000)
for nInit in range(0, len(x10)):
    sol = odeint(RHS2A, [x10[nInit], x20[nInit]], t)
    for nSol in range(0, len(sol)):
        if sol[nSol, 0] > 100.0:
            break
        elif sol[nSol, 1] > 100:
            break
        elif sol[nSol, 0] < -100:
            break
        elif sol[nSol, 1] < -100:
            break
    plt.plot(sol[0:nSol, 0], sol[0:nSol, 1], lw=4.0)
    plt.plot(sol[0, 0], sol[0, 1], 's', ms=20)
    plt.plot(sol[nSol, 0], sol[nSol, 1], 'o', ms=20)

plt.savefig('/home/koorosh/Desktop/Nonlinear_Dynamics/HW/HW2/Report/figure/2E.eps', bbox_inches='tight')
plt.show()
#'''