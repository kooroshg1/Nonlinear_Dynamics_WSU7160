__author__ = 'koorosh'
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 52}

Lambda = 1.0
a = 4.0
def RHS(x1_x2, t):
    x1, x2 = x1_x2
    x1dot = x2
    x2dot = -x1 + Lambda / (a - x1)
    return [x1dot, x2dot]

x1 = np.linspace(-5.0, 5.0, 27)
x2 = np.linspace(-5.0, 5.0, 27)

X1, X2 = np.meshgrid(x1, x2)

normalize = False
X1dot = np.zeros(X1.shape)
X2dot = np.zeros(X2.shape)
for i in range(0,len(X1)):
    for j in range(0, len(X2)):
        X1dot[i, j], X2dot[i, j] = RHS([X1[i, j], X2[i, j]], 0.0)
        if normalize:
            X1dot[i, j] = X1dot[i, j] / (X1dot[i, j]**2 + X2dot[i, j]**2)**0.5
            X2dot[i, j] = X2dot[i, j] / (X1dot[i, j]**2 + X2dot[i, j]**2)**0.5

np.savetxt('X1dot.txt', np.transpose(X1dot))
np.savetxt('X2dot.txt', np.transpose(X2dot))
plt.figure(num=1, figsize=(16, 12))
plt.quiver(X1, X2, X1dot, X2dot)

X10 = [-1.0, -2.0, 0.5, 0.0]
X20 = [-2.0, 2.0, 2.5, -4.0]
plotTrajectory = True
for n in range(0, len(X10)):
    t = np.linspace(0.0, 10.6, 100)
    X1t_X2t = odeint(RHS, [X10[n], X20[n]], t)
    X1t = X1t_X2t[:, 0]
    X2t = X1t_X2t[:, 1]
    if plotTrajectory:
        plt.figure(num=1, figsize=(16, 12))
        plt.plot(X1t, X2t,
                 X1t[0], X2t[0], 'rs',
                 X1t[-1], X2t[-1], 'ro',
                 linewidth=2.0, ms=20.0, mew=2.0)

    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(t, X1t, linewidth=2.0)
    plt.ylabel('Displacement')
    plt.subplot(2,1,2)
    plt.plot(t, X2t, linewidth=2.0)
    plt.ylabel('Velocity')

plt.figure(num=1, figsize=(16, 12))
plt.xlabel('$u_1$')
plt.ylabel('$u_2$')
#plt.xlim([-5.0, 5.0])
plt.ylim([-5.0, 5.0])
plt.rc('font', **font)
plt.gcf().subplots_adjust(bottom=0.15)
filename = str('phase_a' + str(int(a*10.0)) + '_lambda' + str(int(Lambda*10.0)) + '.eps')
#plt.savefig(filename, format='eps')

plt.figure(2)
plt.xlabel('Time')
plt.rc('font', **font)
plt.show()

