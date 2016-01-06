__author__ = 'koorosh'
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
linewidth = 3.0
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 62}
plt.rc('font', **font)
plt.rc('font', **font)


# Define your state function
def F(X, t=0.0):
    x1, x2 = X
    # 1)
    # x1dot = x2
    # x2dot = x1 - x1**3
    # 2)
    # x1dot = x2
    # x2dot = -np.sin(x1)
    # 3)
    x1dot = x1 + 2*x2
    x2dot = 3*x1 + 2*x2
    return [x1dot, x2dot]


# Defines the x1, and x2 axis limits
x1 = np.linspace(-5.0, 5.0, 50)
x2 = np.linspace(-5.0, 5.0, 50)
X1, X2 = np.meshgrid(x1, x2)
X1dot = np.zeros(X1.shape)
X2dot = np.zeros(X2.shape)

# Calculate the vectors
normalize = True
for ni in range(0, len(x1)):
    for nj in range(0, len(x2)):
        X1dot_, X2dot_ = F([X1[ni, nj], X2[ni, nj]])
        if normalize:
            X1dot[ni, nj] = X1dot_ / (X1dot_**2.0 + X2dot_**2.0)**0.5
            X2dot[ni, nj] = X2dot_ / (X1dot_**2.0 + X2dot_**2.0)**0.5

plt.figure(figsize=(20, 20))
plt.quiver(X1, X2, X1dot, X2dot)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([x1[0], x1[-1]])
plt.ylim([x2[0], x2[-1]])

x10 = [-1.5, 1.5, 2.0, -3.0]
x20 = [0.0, -1.0, 1.5, 2.0]
t = np.linspace(0.0, 10.0, 1000)
t = np.linspace(0.0, 10.0, 1000)
for nInit in range(0, len(x10)):
    sol = odeint(F, [x10[nInit], x20[nInit]], t)
    for nSol in range(0, len(sol)):
        if sol[nSol, 0] > 100.0:
            break
        elif sol[nSol, 1] > 100:
            break
        elif sol[nSol, 0] < -100:
            break
        elif sol[nSol, 1] < -100:
            break
    plt.plot(sol[0:nSol, 0], sol[0:nSol, 1], lw=linewidth)
    plt.plot(sol[0, 0], sol[0, 1], 's', ms=10)
    plt.plot(sol[nSol, 0], sol[nSol, 1], 'o', ms=10)

plt.savefig('1.eps', bbox_inches='tight')
# plt.show()

A = np.matrix([[1, 2.],[3.0, 2.0]])
eval, evec = np.linalg.eig(A)
print(eval)
print(evec)
print(np.linalg.det(A))