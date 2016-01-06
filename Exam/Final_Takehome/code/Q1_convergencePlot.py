__author__ = 'koorosh gobal'
import numpy as np
import matplotlib.pylab as plt
import matplotlib
# ----------------------------
linewidth = 3.0
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 42}
plt.rc('font', **font)
linewidth = 3
# ----------------------------
F = np.loadtxt('FConvergence.txt')
X = np.loadtxt('xConvergence.txt')

plt.figure(figsize=(20, 10))
plt.plot(F, 'k',
         lw=linewidth, mew=linewidth)
plt.xlabel('Iteration Number')
plt.ylabel('F')
plt.grid('on')
plt.savefig('F_-2.eps', bbox_inches='tight')

plt.figure(figsize=(20, 10))
plt.plot(X, 'k',
         lw=linewidth, mew=linewidth)
plt.xlabel('Iteration Number')
plt.ylabel('X')
plt.grid('on')
y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.savefig('X_-2.eps', bbox_inches='tight')
plt.show()