__author__ = 'koorosh'
import numpy as np
import matplotlib.pyplot as plt
# # --------------------------------------------------------------------------------------------------------------------
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 52}
plt.rc('font', **font)
linewidth = 9.0
markersize = 20
# # --------------------------------------------------------------------------------------------------------------------
HB_X = np.loadtxt('HB_X.txt')
HB_T = np.loadtxt('HB_t.txt')
ND_X = np.loadtxt('ND_X.txt')
ND_T = np.loadtxt('ND_T.txt')

plt.figure(figsize=(30,15))
plt.plot(HB_T, HB_X, 'k',
         ND_T, ND_X, 'r',
         lw=linewidth, ms=markersize)
plt.legend(['Harmonic Balance', 'Numerical Integration'])
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.savefig('4N9.eps', format='eps', dpi=1000, bbox_inches='tight')
plt.show()