# Python code for Q1 of ME7160
__author__ = 'koorosh gobal'
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------
def FPC(mu0, r=0.1, N=10):
    def F(x, mu):
        return mu - x**2

    def Fx(x, mu):
        return -2 * x

    mu = np.linspace(mu0, 10+mu0, N)
    x1 = np.zeros(N)
    x1[0] = np.sqrt(mu0)
    x2 = np.zeros(N)
    x2[0] = -np.sqrt(mu0)

    for iMu in range(1, len(mu)):
        err = 1
        x1[iMu] = x1[iMu - 1]
        while err > 1e-5:
            dx = -F(x1[iMu], mu[iMu]) / Fx(x1[iMu], mu[iMu])
            x1[iMu] = x1[iMu] + r * dx
            err = F(x1[iMu], mu[iMu])

    for iMu in range(1, len(mu)):
        err = 1
        x2[iMu] = x2[iMu - 1]
        while err > 1e-5:
            dx = -F(x2[iMu], mu[iMu]) / Fx(x2[iMu], mu[iMu])
            x2[iMu] = x2[iMu] + r * dx
            err = F(x2[iMu], mu[iMu])

    return {'x1':x1, 'x2':x2, 'mu':mu}

sol = FPC(0.0001, N=1001)

mu = sol['mu']
x1 = sol['x1']
x2 = sol['x2']

plt.figure()
plt.plot(mu, x1, 'k',
         np.linspace(0, mu[-1], 10), np.sqrt(np.linspace(0, mu[-1], 10)), 
         	'wo',
         mu, x2, 'k--',
         np.linspace(0, mu[-1], 10), -np.sqrt(np.linspace(0, mu[-1], 10)), 
         	'wo')
plt.legend(['Continuation', 'Analytical'],loc='best')
plt.xlabel('$\mu$')
plt.ylabel('x')
plt.show()
