__author__ = 'koorosh'
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------
markersize = 15
linewidth = 3
markeredgewidth = linewidth
fontsize = 42
# ------------------------------------
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : fontsize}

plt.rc('font', **font)
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
        # xConvergence = np.array([x1[iMu - 1]])
        # FConvergence = np.array([F(x1[iMu - 1], mu[iMu])])
        while err > 1e-5:
            dx = -F(x1[iMu], mu[iMu]) / Fx(x1[iMu], mu[iMu])
            x1[iMu] = x1[iMu] + r * dx
            err = F(x1[iMu], mu[iMu])
        #     if np.abs(mu[iMu] - 1) < 0.001:
        #         xConvergence = np.append(xConvergence, x1[iMu])
        #         FConvergence = np.append(FConvergence, err)
        # if np.abs(mu[iMu] - 1) < 0.001:
        #     print(mu[iMu])
        #     np.savetxt('xConvergence.txt', xConvergence)
        #     np.savetxt('FConvergence.txt', FConvergence)

    for iMu in range(1, len(mu)):
        err = 1
        x2[iMu] = x2[iMu - 1]
        xConvergence = np.array([x2[iMu - 1]])
        FConvergence = np.array([F(x2[iMu - 1], mu[iMu])])
        while err > 1e-5:
            dx = -F(x2[iMu], mu[iMu]) / Fx(x2[iMu], mu[iMu])
            x2[iMu] = x2[iMu] + r * dx
            err = F(x2[iMu], mu[iMu])
            if np.abs(mu[iMu] - 2) < 0.001:
                xConvergence = np.append(xConvergence, x2[iMu])
                FConvergence = np.append(FConvergence, err)
        if np.abs(mu[iMu] - 2) < 0.001:
            print(mu[iMu])
            np.savetxt('xConvergence.txt', xConvergence)
            np.savetxt('FConvergence.txt', FConvergence)

    return {'x1':x1, 'x2':x2, 'mu':mu}

sol = FPC(0.0001, N=1001)

mu = sol['mu']
x1 = sol['x1']
x2 = sol['x2']

plt.figure(figsize=(20, 10))
plt.plot(mu, x1, 'k',
         np.linspace(0, mu[-1], 10), np.sqrt(np.linspace(0, mu[-1], 10)), 'wo',
         mu, x2, 'k--',
         np.linspace(0, mu[-1], 10), -np.sqrt(np.linspace(0, mu[-1], 10)), 'wo',
         ms=markersize, lw=linewidth, mew=markeredgewidth)
plt.legend(['Continuation', 'Analytical'],loc='best')
plt.xlabel('$\mu$', fontsize=fontsize)
plt.ylabel('x', fontsize=fontsize)
plt.savefig('bifurcation.eps', bbox_inches='tight')
plt.show()
