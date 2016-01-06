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

hist = np.loadtxt('hist.txt')

plt.figure(figsize=(20, 10))
plt.plot(hist, 'ko', ms=markersize)
plt.xlabel('Number of iterations')
plt.ylabel('Residual value')
plt.ylim([0, 6])
plt.savefig('convergence_study_33.eps', bbox_inches='tight')
plt.show()