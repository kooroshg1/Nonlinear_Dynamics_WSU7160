__author__ = 'koorosh'
import numpy as np
import matplotlib.pyplot as plt
np.seterr(all='ignore')  # Suppressing the runtime waring
# # For report
# linewidth = 12.0
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 52}
plt.rc('font', **font)
linewidth = 9.0
# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 32}
# plt.rc('font', **font)

muStart = -0.000001
muEnd = 0.000001
nMu = 100001
mu = np.linspace(muStart, muEnd, nMu)
delta = 0.001

x01 = (-delta + np.sqrt(delta**2 + 4 * mu)) / 2
ind01 = np.argsort(x01)
x01 = x01[ind01]
mu01 = mu[ind01]
lambda01 = 3 * x01**2 + 2 * delta * x01 - mu01

x02 = (-delta - np.sqrt(delta**2 + 4 * mu)) / 2
ind02 = np.argsort(x02)
x02 = x02[ind02]
mu02 = mu[ind02]
lambda02 = 3 * x02**2 + 2 * delta * x01 - mu02

x0 = np.concatenate((x01, x02), axis=0)
lambda0 = np.concatenate((lambda01, lambda02), axis=0)
mu0 = np.concatenate((mu01, mu02), axis=0)

# np.savetxt('x0.txt', x0, fmt='%2.2f')
# np.savetxt('lambda0.txt', lambda0, fmt='%2.2f')
# np.savetxt('mu0.txt', mu0, fmt='%2.2f')

x0Stable = np.zeros(len(x01)+len(x02))
mu0Stable = np.zeros(len(x01)+len(x02))
x0Unstable = np.zeros(len(x01)+len(x02))
mu0Unstable = np.zeros(len(x01)+len(x02))

for ix0 in range(0, len(x0Stable)):
    if lambda0[ix0] <= 0:
        x0Stable[ix0] = x0[ix0]
        x0Unstable[ix0] = np.nan
        mu0Stable[ix0] = mu0[ix0]
        mu0Unstable[ix0] = np.nan
    elif lambda0[ix0] > 0:
        x0Stable[ix0] = np.nan
        x0Unstable[ix0] = x0[ix0]
        mu0Stable[ix0] = np.nan
        mu0Unstable[ix0] = mu0[ix0]
    else:
        x0Stable[ix0] = np.nan
        x0Unstable[ix0] = np.nan
        mu0Stable[ix0] = np.nan
        mu0Unstable[ix0] = np.nan

np.savetxt('x0stable.txt', x0Stable, fmt='%2.2f')
np.savetxt('mu0stable.txt', mu0Stable, fmt='%2.2f')

figureFileName = '/home/koorosh/Desktop/ME7160/Exam/Midterm_2_Takehome/report/figure/Q2/modified/delta' \
                 + str(delta) + \
                 '_zoomed.eps'

plt.figure(figsize=(30,15))
# plt.figure()
plt.plot(np.linspace(muStart, 0, nMu / 2), np.zeros(nMu / 2), 'k--',
         np.linspace(0, muEnd, nMu / 2), np.zeros(nMu / 2), 'k',
         mu0Stable, x0Stable, 'k',
         mu0Unstable, x0Unstable, 'k--',
         lw=linewidth)
plotTitle = '$\delta$ = ' + str(delta)
# plt.title(plotTitle)
plt.xlabel('$\mu$')
plt.ylabel('$x$')
plt.xlim([muStart, muEnd])
# plt.xlim([-0.01, 0.01])
plt.ylim([-0.001, 0.001])
# plt.ylim([-3, 2])
# plt.grid(b=True, which='major', color='k', linestyle='--')
plt.savefig(figureFileName, format='eps', dpi=1000, bbox_inches='tight')
plt.show()
